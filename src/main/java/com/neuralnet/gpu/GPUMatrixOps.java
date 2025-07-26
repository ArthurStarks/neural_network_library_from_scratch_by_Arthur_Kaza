package com.neuralnet.gpu;

import java.util.logging.Logger;

/**
 * Unified GPU Matrix Operations
 * Automatically selects between CUDA and OpenCL based on availability
 * Provides fallback to CPU operations when GPU is not available
 */
public class GPUMatrixOps {
    private static final Logger logger = Logger.getLogger(GPUMatrixOps.class.getName());
    
    private static boolean initialized = false;
    private static boolean gpuAvailable = false;
    private static boolean useCUDA = false;
    
    /**
     * Initialize GPU matrix operations
     */
    public static void initialize() {
        if (initialized) {
            return;
        }
        
        // Initialize GPU context
        GPUContext.initialize();
        
        if (GPUContext.isGPUAvaliable()) {
            gpuAvailable = true;
            useCUDA = GPUContext.isCUDAvailable();
            
            if (useCUDA) {
                CUDAMatrixOps.initialize();
                logger.info("Using CUDA for GPU acceleration");
            } else {
                OpenCLMatrixOps.initialize();
                logger.info("Using OpenCL for GPU acceleration");
            }
        } else {
            logger.info("GPU not available, using CPU operations");
        }
        
        initialized = true;
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
        
        if (gpuAvailable) {
            if (useCUDA) {
                CUDAMatrixOps.matrixMultiply(A, m, k, B, k, n, C, m, m, n, alpha, beta);
            } else {
                OpenCLMatrixOps.matrixMultiply(A, B, C, m, n, k, alpha, beta);
            }
        } else {
            // CPU fallback
            cpuMatrixMultiply(A, B, C, m, n, k, alpha, beta);
        }
    }
    
    /**
     * Vector addition: y = y + alpha * x
     */
    public static void vectorAdd(double[] x, double[] y, double alpha) {
        if (!initialized) {
            initialize();
        }
        
        if (gpuAvailable) {
            if (useCUDA) {
                CUDAMatrixOps.vectorAdd(x, y, alpha);
            } else {
                OpenCLMatrixOps.vectorAdd(x, y, alpha);
            }
        } else {
            // CPU fallback
            cpuVectorAdd(x, y, alpha);
        }
    }
    
    /**
     * Element-wise multiplication: y = x * y
     */
    public static void elementWiseMultiply(double[] x, double[] y) {
        if (!initialized) {
            initialize();
        }
        
        if (gpuAvailable) {
            if (useCUDA) {
                CUDAMatrixOps.elementWiseMultiply(x, y);
            } else {
                OpenCLMatrixOps.elementWiseMultiply(x, y);
            }
        } else {
            // CPU fallback
            cpuElementWiseMultiply(x, y);
        }
    }
    
    /**
     * Matrix transpose
     */
    public static void matrixTranspose(double[] A, double[] AT, int m, int n) {
        if (!initialized) {
            initialize();
        }
        
        if (gpuAvailable && useCUDA) {
            CUDAMatrixOps.matrixTranspose(A, AT, m, n);
        } else {
            // CPU fallback (OpenCL transpose not implemented)
            cpuMatrixTranspose(A, AT, m, n);
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
        
        if (gpuAvailable) {
            if (useCUDA) {
                CUDAMatrixOps.batchMatrixMultiply(A, B, C, alpha, beta);
            } else {
                OpenCLMatrixOps.batchMatrixMultiply(A, B, C, alpha, beta);
            }
        } else {
            // CPU fallback
            cpuBatchMatrixMultiply(A, B, C, alpha, beta);
        }
    }
    
    /**
     * Check if GPU acceleration is available
     */
    public static boolean isGPUAvaliable() {
        if (!initialized) {
            initialize();
        }
        return gpuAvailable;
    }
    
    /**
     * Check if using CUDA
     */
    public static boolean isUsingCUDA() {
        if (!initialized) {
            initialize();
        }
        return useCUDA;
    }
    
    /**
     * Get GPU information
     */
    public static GPUContext.GPUInfo getGPUInfo() {
        if (!initialized) {
            initialize();
        }
        return GPUContext.getGPUInfo();
    }
    
    // CPU fallback implementations
    
    private static void cpuMatrixMultiply(
            double[] A, double[] B, double[] C,
            int m, int n, int k, double alpha, double beta) {
        
        // Initialize C with beta * C
        for (int i = 0; i < m * n; i++) {
            C[i] *= beta;
        }
        
        // Perform matrix multiplication
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                C[i * n + j] += alpha * sum;
            }
        }
    }
    
    private static void cpuVectorAdd(double[] x, double[] y, double alpha) {
        for (int i = 0; i < x.length; i++) {
            y[i] += alpha * x[i];
        }
    }
    
    private static void cpuElementWiseMultiply(double[] x, double[] y) {
        for (int i = 0; i < x.length; i++) {
            y[i] *= x[i];
        }
    }
    
    private static void cpuMatrixTranspose(double[] A, double[] AT, int m, int n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                AT[j * m + i] = A[i * n + j];
            }
        }
    }
    
    private static void cpuBatchMatrixMultiply(
            double[][][] A, double[][][] B, double[][][] C,
            double alpha, double beta) {
        
        int batchSize = A.length;
        for (int b = 0; b < batchSize; b++) {
            cpuMatrixMultiply(A[b], B[b], C[b], 
                            A[b].length, B[b][0].length, A[b][0].length,
                            alpha, beta);
        }
    }
    
    /**
     * Cleanup GPU resources
     */
    public static void cleanup() {
        if (initialized) {
            if (useCUDA) {
                CUDAMatrixOps.cleanup();
            } else if (gpuAvailable) {
                OpenCLMatrixOps.cleanup();
            }
            GPUContext.cleanup();
            initialized = false;
            gpuAvailable = false;
            useCUDA = false;
            logger.info("GPU matrix operations cleaned up");
        }
    }
} 