package com.neuralnet.gpu;

import com.neuralnet.util.PerformanceMonitor;
import java.util.logging.Logger;

/**
 * Comprehensive GPU Test Suite
 * Tests GPU functionality, performance, and optimization
 */
public class GPUTestSuite {
    private static final Logger logger = Logger.getLogger(GPUTestSuite.class.getName());
    
    /**
     * Run complete GPU test suite
     */
    public static void runAllTests() {
        System.out.println("=== GPU TEST SUITE ===");
        
        testGPUInitialization();
        testGPUMatrixOperations();
        testGPUMemoryManagement();
        testGPUPerformance();
        testGPUScalability();
        testGPUOptimization();
        
        System.out.println("=== GPU TEST SUITE COMPLETE ===");
    }
    
    /**
     * Test GPU initialization and device detection
     */
    public static void testGPUInitialization() {
        System.out.println("\n--- GPU Initialization Test ---");
        
        try {
            // Initialize GPU context
            GPUContext.initialize();
            
            // Check GPU availability
            boolean gpuAvailable = GPUContext.isGPUAvaliable();
            boolean cudaAvailable = GPUContext.isCUDAvailable();
            boolean openclAvailable = GPUContext.isOpenCLAvailable();
            
            System.out.println("GPU Available: " + gpuAvailable);
            System.out.println("CUDA Available: " + cudaAvailable);
            System.out.println("OpenCL Available: " + openclAvailable);
            
            if (gpuAvailable) {
                GPUContext.GPUInfo gpuInfo = GPUContext.getGPUInfo();
                System.out.println("GPU Info: " + gpuInfo);
                System.out.println("GPU Memory: " + (gpuInfo.getGlobalMemory() / (1024 * 1024)) + " MB");
                System.out.println("Compute Units: " + gpuInfo.getComputeUnits());
            }
            
            // Test matrix operations initialization
            GPUMatrixOps.initialize();
            System.out.println("GPU Matrix Operations Initialized: " + GPUMatrixOps.isGPUAvaliable());
            
        } catch (Exception e) {
            System.err.println("GPU Initialization Failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test GPU matrix operations
     */
    public static void testGPUMatrixOperations() {
        System.out.println("\n--- GPU Matrix Operations Test ---");
        
        try {
            // Test matrix multiplication
            int m = 256, n = 256, k = 256;
            double[] A = new double[m * k];
            double[] B = new double[k * n];
            double[] C = new double[m * n];
            
            // Initialize test matrices
            for (int i = 0; i < A.length; i++) A[i] = Math.random();
            for (int i = 0; i < B.length; i++) B[i] = Math.random();
            
            // Test GPU matrix multiplication
            PerformanceMonitor.reset();
            GPUMatrixOps.matrixMultiply(A, B, C, m, n, k, 1.0, 0.0);
            double gpuTime = PerformanceMonitor.getTotalTime("matrix_multiply");
            
            System.out.println("GPU Matrix Multiplication (" + m + "x" + n + "): " + gpuTime + " ms");
            
            // Test vector operations
            double[] x = new double[1000];
            double[] y = new double[1000];
            for (int i = 0; i < x.length; i++) {
                x[i] = Math.random();
                y[i] = Math.random();
            }
            
            PerformanceMonitor.reset();
            GPUMatrixOps.vectorAdd(x, y, 0.5);
            double vectorTime = PerformanceMonitor.getTotalTime("vector_add");
            
            System.out.println("GPU Vector Addition: " + vectorTime + " ms");
            
            // Test element-wise operations
            PerformanceMonitor.reset();
            GPUMatrixOps.elementWiseMultiply(x, y);
            double elementTime = PerformanceMonitor.getTotalTime("element_wise_multiply");
            
            System.out.println("GPU Element-wise Multiplication: " + elementTime + " ms");
            
        } catch (Exception e) {
            System.err.println("GPU Matrix Operations Test Failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test GPU memory management
     */
    public static void testGPUMemoryManagement() {
        System.out.println("\n--- GPU Memory Management Test ---");
        
        try {
            // Test memory allocation and deallocation
            int[] sizes = {1024, 2048, 4096, 8192};
            
            for (int size : sizes) {
                double[] data = new double[size];
                for (int i = 0; i < size; i++) {
                    data[i] = Math.random();
                }
                
                PerformanceMonitor.reset();
                GPUMatrixOps.matrixMultiply(data, data, data, 
                                          (int)Math.sqrt(size), (int)Math.sqrt(size), (int)Math.sqrt(size), 
                                          1.0, 0.0);
                double time = PerformanceMonitor.getTotalTime("matrix_multiply");
                
                System.out.println("GPU Memory Test (" + size + " elements): " + time + " ms");
            }
            
            // Test memory usage
            double initialMemory = PerformanceMonitor.getCurrentMemoryMB();
            System.out.println("Initial Memory: " + initialMemory + " MB");
            
            // Perform multiple operations
            for (int i = 0; i < 100; i++) {
                double[] A = new double[1024];
                double[] B = new double[1024];
                double[] C = new double[1024];
                
                GPUMatrixOps.matrixMultiply(A, B, C, 32, 32, 32, 1.0, 0.0);
                
                if (i % 20 == 0) {
                    PerformanceMonitor.updateMemoryUsage();
                }
            }
            
            double peakMemory = PerformanceMonitor.getPeakMemoryMB();
            System.out.println("Peak Memory: " + peakMemory + " MB");
            System.out.println("Memory Increase: " + (peakMemory - initialMemory) + " MB");
            
        } catch (Exception e) {
            System.err.println("GPU Memory Management Test Failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test GPU performance vs CPU
     */
    public static void testGPUPerformance() {
        System.out.println("\n--- GPU Performance Test ---");
        
        try {
            int[] sizes = {64, 128, 256, 512, 1024};
            
            for (int size : sizes) {
                double[] A = new double[size * size];
                double[] B = new double[size * size];
                double[] C = new double[size * size];
                
                // Initialize matrices
                for (int i = 0; i < A.length; i++) {
                    A[i] = Math.random();
                    B[i] = Math.random();
                }
                
                // Test GPU performance
                PerformanceMonitor.reset();
                GPUMatrixOps.matrixMultiply(A, B, C, size, size, size, 1.0, 0.0);
                double gpuTime = PerformanceMonitor.getTotalTime("matrix_multiply");
                
                // Test CPU performance (fallback)
                PerformanceMonitor.reset();
                cpuMatrixMultiply(A, B, C, size, size, size, 1.0, 0.0);
                double cpuTime = PerformanceMonitor.getTotalTime("cpu_matrix_multiply");
                
                double speedup = cpuTime / gpuTime;
                System.out.printf("Size %dx%d: GPU=%.3fms, CPU=%.3fms, Speedup=%.2fx\n", 
                                size, size, gpuTime, cpuTime, speedup);
            }
            
        } catch (Exception e) {
            System.err.println("GPU Performance Test Failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test GPU scalability
     */
    public static void testGPUScalability() {
        System.out.println("\n--- GPU Scalability Test ---");
        
        try {
            // Test with different batch sizes
            int[] batchSizes = {1, 4, 8, 16, 32, 64, 128};
            int matrixSize = 256;
            
            for (int batchSize : batchSizes) {
                double[][][] A = new double[batchSize][matrixSize][matrixSize];
                double[][][] B = new double[batchSize][matrixSize][matrixSize];
                double[][][] C = new double[batchSize][matrixSize][matrixSize];
                
                // Initialize batch matrices
                for (int b = 0; b < batchSize; b++) {
                    for (int i = 0; i < matrixSize; i++) {
                        for (int j = 0; j < matrixSize; j++) {
                            A[b][i][j] = Math.random();
                            B[b][i][j] = Math.random();
                        }
                    }
                }
                
                PerformanceMonitor.reset();
                GPUMatrixOps.batchMatrixMultiply(A, B, C, 1.0, 0.0);
                double time = PerformanceMonitor.getTotalTime("batch_matrix_multiply");
                
                double throughput = (batchSize * matrixSize * matrixSize * matrixSize) / (time / 1000.0);
                System.out.printf("Batch Size %d: %.3fms, Throughput=%.0f ops/sec\n", 
                                batchSize, time, throughput);
            }
            
        } catch (Exception e) {
            System.err.println("GPU Scalability Test Failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Test GPU optimization techniques
     */
    public static void testGPUOptimization() {
        System.out.println("\n--- GPU Optimization Test ---");
        
        try {
            // Test different optimization strategies
            int matrixSize = 512;
            double[] A = new double[matrixSize * matrixSize];
            double[] B = new double[matrixSize * matrixSize];
            double[] C = new double[matrixSize * matrixSize];
            
            // Initialize matrices
            for (int i = 0; i < A.length; i++) {
                A[i] = Math.random();
                B[i] = Math.random();
            }
            
            // Test single operation
            PerformanceMonitor.reset();
            GPUMatrixOps.matrixMultiply(A, B, C, matrixSize, matrixSize, matrixSize, 1.0, 0.0);
            double singleTime = PerformanceMonitor.getTotalTime("matrix_multiply");
            
            // Test multiple operations (potential for optimization)
            PerformanceMonitor.reset();
            for (int i = 0; i < 10; i++) {
                GPUMatrixOps.matrixMultiply(A, B, C, matrixSize, matrixSize, matrixSize, 1.0, 0.0);
            }
            double multipleTime = PerformanceMonitor.getTotalTime("matrix_multiply");
            
            System.out.println("Single Operation: " + singleTime + " ms");
            System.out.println("Multiple Operations: " + multipleTime + " ms");
            System.out.println("Average per Operation: " + (multipleTime / 10.0) + " ms");
            System.out.println("Optimization Potential: " + (singleTime - (multipleTime / 10.0)) + " ms");
            
        } catch (Exception e) {
            System.err.println("GPU Optimization Test Failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * CPU fallback matrix multiplication for comparison
     */
    private static void cpuMatrixMultiply(double[] A, double[] B, double[] C, 
                                        int m, int n, int k, double alpha, double beta) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("cpu_matrix_multiply")) {
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
    }
    
    /**
     * Cleanup GPU resources
     */
    public static void cleanup() {
        try {
            GPUMatrixOps.cleanup();
            GPUContext.cleanup();
            System.out.println("GPU resources cleaned up successfully");
        } catch (Exception e) {
            System.err.println("GPU cleanup failed: " + e.getMessage());
        }
    }
} 