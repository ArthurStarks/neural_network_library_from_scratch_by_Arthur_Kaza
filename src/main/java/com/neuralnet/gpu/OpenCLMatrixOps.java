package com.neuralnet.gpu;

import org.jocl.*;
import java.util.logging.Logger;

/**
 * OpenCL-accelerated matrix operations for neural networks
 * Provides GPU acceleration for systems without CUDA support
 */
public class OpenCLMatrixOps {
    private static final Logger logger = Logger.getLogger(OpenCLMatrixOps.class.getName());
    
    private static boolean initialized = false;
    private static cl_program program;
    private static cl_kernel matrixMultiplyKernel;
    private static cl_kernel vectorAddKernel;
    private static cl_kernel elementWiseMultiplyKernel;
    
    // OpenCL kernel source code
    private static final String KERNEL_SOURCE = 
        "__kernel void matrix_multiply(__global const double* A, __global const double* B, __global double* C, " +
        "int m, int n, int k, double alpha, double beta) {\n" +
        "    int row = get_global_id(0);\n" +
        "    int col = get_global_id(1);\n" +
        "    if (row < m && col < n) {\n" +
        "        double sum = 0.0;\n" +
        "        for (int i = 0; i < k; i++) {\n" +
        "            sum += A[row * k + i] * B[i * n + col];\n" +
        "        }\n" +
        "        C[row * n + col] = alpha * sum + beta * C[row * n + col];\n" +
        "    }\n" +
        "}\n" +
        "\n" +
        "__kernel void vector_add(__global const double* x, __global double* y, double alpha, int n) {\n" +
        "    int i = get_global_id(0);\n" +
        "    if (i < n) {\n" +
        "        y[i] += alpha * x[i];\n" +
        "    }\n" +
        "}\n" +
        "\n" +
        "__kernel void element_wise_multiply(__global const double* x, __global double* y, int n) {\n" +
        "    int i = get_global_id(0);\n" +
        "    if (i < n) {\n" +
        "        y[i] *= x[i];\n" +
        "    }\n" +
        "}\n";
    
    /**
     * Initialize OpenCL program and kernels
     */
    public static void initialize() {
        if (initialized) {
            return;
        }
        
        if (!GPUContext.isOpenCLAvailable()) {
            throw new RuntimeException("OpenCL not available");
        }
        
        try {
            // Create program
            program = CL.clCreateProgramWithSource(GPUContext.getOpenCLContext(), 1, 
                                                 new String[]{KERNEL_SOURCE}, null, null);
            
            // Build program
            CL.clBuildProgram(program, 0, null, null, null, null);
            
            // Create kernels
            matrixMultiplyKernel = CL.clCreateKernel(program, "matrix_multiply");
            vectorAddKernel = CL.clCreateKernel(program, "vector_add");
            elementWiseMultiplyKernel = CL.clCreateKernel(program, "element_wise_multiply");
            
            initialized = true;
            logger.info("OpenCL kernels initialized successfully");
            
        } catch (Exception e) {
            logger.severe("Failed to initialize OpenCL kernels: " + e.getMessage());
            throw new RuntimeException("OpenCL initialization failed", e);
        }
    }
    
    /**
     * Matrix multiplication: C = alpha * A * B + beta * C
     */
    public static void matrixMultiply(
            double[] A, double[] B, double[] C,
            int m, int n, int k, double alpha, double beta) {
        
        if (!initialized) {
            initialize();
        }
        
        // Allocate GPU memory
        cl_mem dA = CL.clCreateBuffer(GPUContext.getOpenCLContext(), CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                                     Sizeof.cl_double * m * k, Pointer.to(A), null);
        cl_mem dB = CL.clCreateBuffer(GPUContext.getOpenCLContext(), CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                                     Sizeof.cl_double * k * n, Pointer.to(B), null);
        cl_mem dC = CL.clCreateBuffer(GPUContext.getOpenCLContext(), CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                                     Sizeof.cl_double * m * n, Pointer.to(C), null);
        
        try {
            // Set kernel arguments
            CL.clSetKernelArg(matrixMultiplyKernel, 0, Sizeof.cl_mem, Pointer.to(dA));
            CL.clSetKernelArg(matrixMultiplyKernel, 1, Sizeof.cl_mem, Pointer.to(dB));
            CL.clSetKernelArg(matrixMultiplyKernel, 2, Sizeof.cl_mem, Pointer.to(dC));
            CL.clSetKernelArg(matrixMultiplyKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{m}));
            CL.clSetKernelArg(matrixMultiplyKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));
            CL.clSetKernelArg(matrixMultiplyKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{k}));
            CL.clSetKernelArg(matrixMultiplyKernel, 6, Sizeof.cl_double, Pointer.to(new double[]{alpha}));
            CL.clSetKernelArg(matrixMultiplyKernel, 7, Sizeof.cl_double, Pointer.to(new double[]{beta}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{m, n};
            CL.clEnqueueNDRangeKernel(GPUContext.getOpenCLCommandQueue(), matrixMultiplyKernel, 2, null,
                                     globalWorkSize, null, 0, null, null);
            
            // Read result
            CL.clEnqueueReadBuffer(GPUContext.getOpenCLCommandQueue(), dC, CL.CL_TRUE, 0,
                                  Sizeof.cl_double * m * n, Pointer.to(C), 0, null, null);
            
        } finally {
            // Free GPU memory
            CL.clReleaseMemObject(dA);
            CL.clReleaseMemObject(dB);
            CL.clReleaseMemObject(dC);
        }
    }
    
    /**
     * Vector addition: y = y + alpha * x
     */
    public static void vectorAdd(double[] x, double[] y, double alpha) {
        if (!initialized) {
            initialize();
        }
        
        int n = x.length;
        
        // Allocate GPU memory
        cl_mem dx = CL.clCreateBuffer(GPUContext.getOpenCLContext(), CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                                     Sizeof.cl_double * n, Pointer.to(x), null);
        cl_mem dy = CL.clCreateBuffer(GPUContext.getOpenCLContext(), CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                                     Sizeof.cl_double * n, Pointer.to(y), null);
        
        try {
            // Set kernel arguments
            CL.clSetKernelArg(vectorAddKernel, 0, Sizeof.cl_mem, Pointer.to(dx));
            CL.clSetKernelArg(vectorAddKernel, 1, Sizeof.cl_mem, Pointer.to(dy));
            CL.clSetKernelArg(vectorAddKernel, 2, Sizeof.cl_double, Pointer.to(new double[]{alpha}));
            CL.clSetKernelArg(vectorAddKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{n};
            CL.clEnqueueNDRangeKernel(GPUContext.getOpenCLCommandQueue(), vectorAddKernel, 1, null,
                                     globalWorkSize, null, 0, null, null);
            
            // Read result
            CL.clEnqueueReadBuffer(GPUContext.getOpenCLCommandQueue(), dy, CL.CL_TRUE, 0,
                                  Sizeof.cl_double * n, Pointer.to(y), 0, null, null);
            
        } finally {
            // Free GPU memory
            CL.clReleaseMemObject(dx);
            CL.clReleaseMemObject(dy);
        }
    }
    
    /**
     * Element-wise multiplication: y = x * y
     */
    public static void elementWiseMultiply(double[] x, double[] y) {
        if (!initialized) {
            initialize();
        }
        
        int n = x.length;
        
        // Allocate GPU memory
        cl_mem dx = CL.clCreateBuffer(GPUContext.getOpenCLContext(), CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR,
                                     Sizeof.cl_double * n, Pointer.to(x), null);
        cl_mem dy = CL.clCreateBuffer(GPUContext.getOpenCLContext(), CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR,
                                     Sizeof.cl_double * n, Pointer.to(y), null);
        
        try {
            // Set kernel arguments
            CL.clSetKernelArg(elementWiseMultiplyKernel, 0, Sizeof.cl_mem, Pointer.to(dx));
            CL.clSetKernelArg(elementWiseMultiplyKernel, 1, Sizeof.cl_mem, Pointer.to(dy));
            CL.clSetKernelArg(elementWiseMultiplyKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{n};
            CL.clEnqueueNDRangeKernel(GPUContext.getOpenCLCommandQueue(), elementWiseMultiplyKernel, 1, null,
                                     globalWorkSize, null, 0, null, null);
            
            // Read result
            CL.clEnqueueReadBuffer(GPUContext.getOpenCLCommandQueue(), dy, CL.CL_TRUE, 0,
                                  Sizeof.cl_double * n, Pointer.to(y), 0, null, null);
            
        } finally {
            // Free GPU memory
            CL.clReleaseMemObject(dx);
            CL.clReleaseMemObject(dy);
        }
    }
    
    /**
     * Matrix transpose
     */
    public static void matrixTranspose(double[] A, double[] AT, int m, int n) {
        // For transpose, we'll use CPU implementation for simplicity
        // OpenCL transpose would require a custom kernel
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                AT[j * m + i] = A[i * n + j];
            }
        }
    }
    
    /**
     * Batch matrix multiplication for multiple samples
     */
    public static void batchMatrixMultiply(
            double[][][] A, double[][][] B, double[][][] C,
            double alpha, double beta) {
        
        int batchSize = A.length;
        
        // Process each batch element
        for (int b = 0; b < batchSize; b++) {
            matrixMultiply(A[b], B[b], C[b], 
                         A[b].length, B[b][0].length, A[b][0].length,
                         alpha, beta);
        }
    }
    
    /**
     * Cleanup OpenCL resources
     */
    public static void cleanup() {
        if (initialized) {
            if (matrixMultiplyKernel != null) {
                CL.clReleaseKernel(matrixMultiplyKernel);
                matrixMultiplyKernel = null;
            }
            if (vectorAddKernel != null) {
                CL.clReleaseKernel(vectorAddKernel);
                vectorAddKernel = null;
            }
            if (elementWiseMultiplyKernel != null) {
                CL.clReleaseKernel(elementWiseMultiplyKernel);
                elementWiseMultiplyKernel = null;
            }
            if (program != null) {
                CL.clReleaseProgram(program);
                program = null;
            }
            initialized = false;
            logger.info("OpenCL resources cleaned up");
        }
    }
} 