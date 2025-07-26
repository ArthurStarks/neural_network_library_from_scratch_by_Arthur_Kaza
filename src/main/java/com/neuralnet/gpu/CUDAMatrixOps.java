package com.neuralnet.gpu;

import org.jcuda.*;
import org.jcuda.runtime.*;
import org.jcuda.runtime.JCuda;
import org.jcuda.runtime.cudaMemcpyKind;
import org.jcuda.jcublas.*;
import org.jcuda.jcublas.JCublas;

import java.util.logging.Logger;

/**
 * CUDA-accelerated matrix operations for neural networks
 * Provides high-performance matrix multiplication and element-wise operations
 */
public class CUDAMatrixOps {
    private static final Logger logger = Logger.getLogger(CUDAMatrixOps.class.getName());
    
    private static boolean initialized = false;
    private static cublasHandle cublasHandle;
    
    /**
     * Initialize CUDA BLAS library
     */
    public static void initialize() {
        if (initialized) {
            return;
        }
        
        if (!GPUContext.isCUDAvailable()) {
            throw new RuntimeException("CUDA not available");
        }
        
        cublasHandle = new cublasHandle();
        JCublas.cublasCreate(cublasHandle);
        initialized = true;
        
        logger.info("CUDA BLAS initialized successfully");
    }
    
    /**
     * Matrix multiplication: C = alpha * A * B + beta * C
     */
    public static void matrixMultiply(
            double[] A, int lda, int m, int k,
            double[] B, int ldb, int k, int n,
            double[] C, int ldc, int m, int n,
            double alpha, double beta) {
        
        if (!initialized) {
            initialize();
        }
        
        // Allocate GPU memory
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        
        JCuda.cudaMalloc(dA, m * k * Sizeof.DOUBLE);
        JCuda.cudaMalloc(dB, k * n * Sizeof.DOUBLE);
        JCuda.cudaMalloc(dC, m * n * Sizeof.DOUBLE);
        
        try {
            // Copy data to GPU
            JCuda.cudaMemcpy(dA, Pointer.to(A), m * k * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaMemcpy(dB, Pointer.to(B), k * n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaMemcpy(dC, Pointer.to(C), m * n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            
            // Perform matrix multiplication
            JCublas.cublasDgemm(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
                               m, n, k, Pointer.to(new double[]{alpha}),
                               dA, lda, dB, ldb, Pointer.to(new double[]{beta}), dC, ldc);
            
            // Copy result back to CPU
            JCuda.cudaMemcpy(Pointer.to(C), dC, m * n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            
        } finally {
            // Free GPU memory
            JCuda.cudaFree(dA);
            JCuda.cudaFree(dB);
            JCuda.cudaFree(dC);
        }
    }
    
    /**
     * Vector addition: y = alpha * x + y
     */
    public static void vectorAdd(double[] x, double[] y, double alpha) {
        if (!initialized) {
            initialize();
        }
        
        int n = x.length;
        
        // Allocate GPU memory
        Pointer dx = new Pointer();
        Pointer dy = new Pointer();
        
        JCuda.cudaMalloc(dx, n * Sizeof.DOUBLE);
        JCuda.cudaMalloc(dy, n * Sizeof.DOUBLE);
        
        try {
            // Copy data to GPU
            JCuda.cudaMemcpy(dx, Pointer.to(x), n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaMemcpy(dy, Pointer.to(y), n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            
            // Perform vector addition
            JCublas.cublasDaxpy(cublasHandle, n, Pointer.to(new double[]{alpha}), dx, 1, dy, 1);
            
            // Copy result back to CPU
            JCuda.cudaMemcpy(Pointer.to(y), dy, n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            
        } finally {
            // Free GPU memory
            JCuda.cudaFree(dx);
            JCuda.cudaFree(dy);
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
        Pointer dx = new Pointer();
        Pointer dy = new Pointer();
        
        JCuda.cudaMalloc(dx, n * Sizeof.DOUBLE);
        JCuda.cudaMalloc(dy, n * Sizeof.DOUBLE);
        
        try {
            // Copy data to GPU
            JCuda.cudaMemcpy(dx, Pointer.to(x), n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            JCuda.cudaMemcpy(dy, Pointer.to(y), n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            
            // Perform element-wise multiplication using custom kernel
            // Note: This would require a custom CUDA kernel for optimal performance
            // For now, we'll use CPU fallback
            for (int i = 0; i < n; i++) {
                y[i] *= x[i];
            }
            
        } finally {
            // Free GPU memory
            JCuda.cudaFree(dx);
            JCuda.cudaFree(dy);
        }
    }
    
    /**
     * Matrix transpose
     */
    public static void matrixTranspose(double[] A, double[] AT, int m, int n) {
        if (!initialized) {
            initialize();
        }
        
        // Allocate GPU memory
        Pointer dA = new Pointer();
        Pointer dAT = new Pointer();
        
        JCuda.cudaMalloc(dA, m * n * Sizeof.DOUBLE);
        JCuda.cudaMalloc(dAT, n * m * Sizeof.DOUBLE);
        
        try {
            // Copy data to GPU
            JCuda.cudaMemcpy(dA, Pointer.to(A), m * n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            
            // Perform matrix transpose
            JCublas.cublasDgeam(cublasHandle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,
                               n, m, Pointer.to(new double[]{1.0}), dA, m,
                               Pointer.to(new double[]{0.0}), dAT, n, dAT, n);
            
            // Copy result back to CPU
            JCuda.cudaMemcpy(Pointer.to(AT), dAT, n * m * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            
        } finally {
            // Free GPU memory
            JCuda.cudaFree(dA);
            JCuda.cudaFree(dAT);
        }
    }
    
    /**
     * Batch matrix multiplication for multiple samples
     */
    public static void batchMatrixMultiply(
            double[][][] A, double[][][] B, double[][][] C,
            double alpha, double beta) {
        
        if (!initialized) {
            initialize();
        }
        
        int batchSize = A.length;
        int m = A[0].length;
        int k = A[0][0].length;
        int n = B[0][0].length;
        
        // Allocate GPU memory for batch
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();
        
        JCuda.cudaMalloc(dA, batchSize * m * k * Sizeof.DOUBLE);
        JCuda.cudaMalloc(dB, batchSize * k * n * Sizeof.DOUBLE);
        JCuda.cudaMalloc(dC, batchSize * m * n * Sizeof.DOUBLE);
        
        try {
            // Copy batch data to GPU
            for (int b = 0; b < batchSize; b++) {
                JCuda.cudaMemcpy(dA.withByteOffset(b * m * k * Sizeof.DOUBLE),
                                Pointer.to(A[b]), m * k * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
                JCuda.cudaMemcpy(dB.withByteOffset(b * k * n * Sizeof.DOUBLE),
                                Pointer.to(B[b]), k * n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyHostToDevice);
            }
            
            // Perform batch matrix multiplication
            for (int b = 0; b < batchSize; b++) {
                JCublas.cublasDgemm(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
                                   m, n, k, Pointer.to(new double[]{alpha}),
                                   dA.withByteOffset(b * m * k * Sizeof.DOUBLE), m,
                                   dB.withByteOffset(b * k * n * Sizeof.DOUBLE), k,
                                   Pointer.to(new double[]{beta}),
                                   dC.withByteOffset(b * m * n * Sizeof.DOUBLE), m);
            }
            
            // Copy results back to CPU
            for (int b = 0; b < batchSize; b++) {
                JCuda.cudaMemcpy(Pointer.to(C[b]), dC.withByteOffset(b * m * n * Sizeof.DOUBLE),
                                m * n * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);
            }
            
        } finally {
            // Free GPU memory
            JCuda.cudaFree(dA);
            JCuda.cudaFree(dB);
            JCuda.cudaFree(dC);
        }
    }
    
    /**
     * Cleanup CUDA resources
     */
    public static void cleanup() {
        if (initialized && cublasHandle != null) {
            JCublas.cublasDestroy(cublasHandle);
            cublasHandle = null;
            initialized = false;
            logger.info("CUDA BLAS resources cleaned up");
        }
    }
} 