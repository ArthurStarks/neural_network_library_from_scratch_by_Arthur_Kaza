package com.neuralnet.gpu;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.AfterAll;

/**
 * Comprehensive GPU Optimization Test
 * Tests all GPU optimization features: testing, memory pooling, batch optimization
 */
public class GPUOptimizationTest {
    
    @BeforeAll
    public static void setup() {
        System.out.println("🚀 Initializing GPU Optimization Test Suite");
        
        // Initialize GPU context
        try {
            GPUContext.initialize();
            GPUMatrixOps.initialize();
            System.out.println("✅ GPU initialization successful");
        } catch (Exception e) {
            System.out.println("⚠️  GPU initialization failed: " + e.getMessage());
            System.out.println("   Continuing with CPU fallback");
        }
    }
    
    @Test
    public void testGPUFunctionality() {
        System.out.println("\n=== Testing GPU Functionality ===");
        GPUTestSuite.runAllTests();
    }
    
    @Test
    public void testMemoryPooling() {
        System.out.println("\n=== Testing Memory Pooling ===");
        
        // Test memory pool allocation and deallocation
        int[] sizes = {1024, 2048, 4096, 8192};
        
        for (int size : sizes) {
            System.out.println("Testing memory pool with size: " + size + " bytes");
            
            if (GPUContext.isCUDAvailable()) {
                // Test CUDA memory pooling
                for (int i = 0; i < 10; i++) {
                    Pointer ptr = GPUMemoryPool.allocateCUDAMemory(size);
                    GPUMemoryPool.freeCUDAMemory(ptr, size);
                }
            } else if (GPUContext.isOpenCLAvailable()) {
                // Test OpenCL memory pooling
                for (int i = 0; i < 10; i++) {
                    cl_mem mem = GPUMemoryPool.allocateOpenCLMemory(size, CL.CL_MEM_READ_WRITE);
                    GPUMemoryPool.freeOpenCLMemory(mem, size);
                }
            }
        }
        
        // Print memory pool statistics
        GPUMemoryPool.MemoryPoolStats stats = GPUMemoryPool.getStats();
        System.out.println("Memory Pool Statistics: " + stats);
        
        // Optimize pools
        GPUMemoryPool.optimizePools();
        System.out.println("Memory pools optimized");
    }
    
    @Test
    public void testBatchSizeOptimization() {
        System.out.println("\n=== Testing Batch Size Optimization ===");
        
        // Test different network architectures
        int[][] architectures = {
            {784, 128, 10},   // MNIST-like
            {784, 256, 10},   // Larger hidden layer
            {256, 128, 64}    // Small network
        };
        
        for (int[] arch : architectures) {
            System.out.println("\nOptimizing batch size for: " + arch[0] + " -> " + arch[1] + " -> " + arch[2]);
            
            GPUBatchOptimizer.BatchOptimizationResult result = 
                GPUBatchOptimizer.optimizeBatchSize(arch[0], arch[1], arch[2]);
            
            System.out.println("Best batch size: " + result.getBatchSize());
            System.out.println("Throughput: " + String.format("%.2f", result.getThroughput()) + " samples/sec");
            System.out.println("Memory usage: " + String.format("%.2f", result.getMemoryUsage()) + " MB");
        }
    }
    
    @Test
    public void testCompleteOptimization() {
        System.out.println("\n=== Running Complete GPU Optimization ===");
        GPUOptimizationRunner.runCompleteOptimization();
    }
    
    @Test
    public void testGPUNetworkPerformance() {
        System.out.println("\n=== Testing GPU Network Performance ===");
        
        // Create test data
        int inputSize = 784;
        int hiddenSize = 256;
        int outputSize = 10;
        int batchSize = 64;
        
        double[][] inputs = new double[batchSize][inputSize];
        double[][] targets = new double[batchSize][outputSize];
        
        // Initialize with random data
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                inputs[i][j] = Math.random();
            }
            for (int j = 0; j < outputSize; j++) {
                targets[i][j] = Math.random();
            }
        }
        
        // Test GPU matrix operations performance
        System.out.println("Testing GPU matrix operations...");
        
        // Forward pass simulation
        for (int i = 0; i < batchSize; i++) {
            double[] hidden = new double[hiddenSize];
            double[] output = new double[outputSize];
            
            // Input -> Hidden
            GPUMatrixOps.matrixMultiply(inputs[i], new double[inputSize * hiddenSize], 
                                      hidden, 1, hiddenSize, inputSize, 1.0, 0.0);
            
            // Hidden -> Output
            GPUMatrixOps.matrixMultiply(hidden, new double[hiddenSize * outputSize], 
                                      output, 1, outputSize, hiddenSize, 1.0, 0.0);
        }
        
        System.out.println("GPU matrix operations completed successfully");
    }
    
    @Test
    public void testMemoryEfficiency() {
        System.out.println("\n=== Testing Memory Efficiency ===");
        
        double initialMemory = com.neuralnet.util.PerformanceMonitor.getCurrentMemoryMB();
        System.out.println("Initial memory: " + initialMemory + " MB");
        
        // Perform intensive operations
        for (int i = 0; i < 100; i++) {
            double[] A = new double[1024];
            double[] B = new double[1024];
            double[] C = new double[1024];
            
            // Initialize with random data
            for (int j = 0; j < A.length; j++) {
                A[j] = Math.random();
                B[j] = Math.random();
            }
            
            // Perform matrix multiplication
            GPUMatrixOps.matrixMultiply(A, B, C, 32, 32, 32, 1.0, 0.0);
            
            // Update memory usage every 20 operations
            if (i % 20 == 0) {
                com.neuralnet.util.PerformanceMonitor.updateMemoryUsage();
            }
        }
        
        double peakMemory = com.neuralnet.util.PerformanceMonitor.getPeakMemoryMB();
        System.out.println("Peak memory: " + peakMemory + " MB");
        System.out.println("Memory increase: " + (peakMemory - initialMemory) + " MB");
        
        // Get memory pool statistics
        GPUMemoryPool.MemoryPoolStats stats = GPUMemoryPool.getStats();
        System.out.println("Memory pool hit rate: " + String.format("%.2f", stats.getHitRate() * 100) + "%");
    }
    
    @Test
    public void testScalability() {
        System.out.println("\n=== Testing GPU Scalability ===");
        
        // Test different matrix sizes
        int[] sizes = {64, 128, 256, 512};
        
        for (int size : sizes) {
            System.out.println("Testing matrix size: " + size + "x" + size);
            
            double[] A = new double[size * size];
            double[] B = new double[size * size];
            double[] C = new double[size * size];
            
            // Initialize matrices
            for (int i = 0; i < A.length; i++) {
                A[i] = Math.random();
                B[i] = Math.random();
            }
            
            // Measure GPU performance
            long startTime = System.nanoTime();
            GPUMatrixOps.matrixMultiply(A, B, C, size, size, size, 1.0, 0.0);
            long endTime = System.nanoTime();
            
            double gpuTime = (endTime - startTime) / 1_000_000.0; // Convert to milliseconds
            System.out.printf("  GPU time: %.3f ms\n", gpuTime);
            
            // Calculate throughput
            double operations = size * size * size * 2.0; // FLOPS for matrix multiplication
            double throughput = operations / (gpuTime / 1000.0); // Operations per second
            System.out.printf("  Throughput: %.0f operations/sec\n", throughput);
        }
    }
    
    @AfterAll
    public static void cleanup() {
        System.out.println("\n🧹 Cleaning up GPU resources...");
        GPUOptimizationRunner.cleanup();
        System.out.println("✅ GPU optimization test suite completed");
    }
} 