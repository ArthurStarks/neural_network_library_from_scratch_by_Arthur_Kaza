package com.neuralnet.gpu;

import com.neuralnet.util.PerformanceMonitor;
import java.util.logging.Logger;

/**
 * Comprehensive GPU Optimization Runner
 * Executes all GPU optimization steps: testing, memory pooling, batch optimization
 */
public class GPUOptimizationRunner {
    private static final Logger logger = Logger.getLogger(GPUOptimizationRunner.class.getName());
    
    /**
     * Run complete GPU optimization suite
     */
    public static void runCompleteOptimization() {
        System.out.println("🚀 === COMPLETE GPU OPTIMIZATION SUITE ===");
        
        // Step 1: Test GPU functionality
        System.out.println("\n📋 Step 1: GPU Functionality Testing");
        GPUTestSuite.runAllTests();
        
        // Step 2: Memory pool optimization
        System.out.println("\n💾 Step 2: Memory Pool Optimization");
        optimizeMemoryPools();
        
        // Step 3: Batch size optimization
        System.out.println("\n⚡ Step 3: Batch Size Optimization");
        optimizeBatchSizes();
        
        // Step 4: Performance benchmarking
        System.out.println("\n📊 Step 4: Performance Benchmarking");
        runPerformanceBenchmarks();
        
        // Step 5: Generate optimization report
        System.out.println("\n📋 Step 5: Optimization Report");
        generateOptimizationReport();
        
        System.out.println("\n✅ === GPU OPTIMIZATION COMPLETE ===");
    }
    
    /**
     * Optimize GPU memory pools
     */
    private static void optimizeMemoryPools() {
        System.out.println("--- Memory Pool Optimization ---");
        
        // Test memory pool performance
        PerformanceMonitor.reset();
        
        // Allocate and free memory multiple times
        int[] sizes = {1024, 2048, 4096, 8192, 16384};
        for (int i = 0; i < 100; i++) {
            for (int size : sizes) {
                if (GPUContext.isCUDAvailable()) {
                    Pointer ptr = GPUMemoryPool.allocateCUDAMemory(size);
                    GPUMemoryPool.freeCUDAMemory(ptr, size);
                } else if (GPUContext.isOpenCLAvailable()) {
                    cl_mem mem = GPUMemoryPool.allocateOpenCLMemory(size, CL.CL_MEM_READ_WRITE);
                    GPUMemoryPool.freeOpenCLMemory(mem, size);
                }
            }
        }
        
        // Get memory pool statistics
        GPUMemoryPool.MemoryPoolStats stats = GPUMemoryPool.getStats();
        System.out.println("Memory Pool Statistics:");
        System.out.println("  Total Allocated: " + stats.getTotalAllocated() + " bytes");
        System.out.println("  Total Freed: " + stats.getTotalFreed() + " bytes");
        System.out.println("  Current Usage: " + stats.getCurrentUsage() + " bytes");
        System.out.println("  Pool Hits: " + stats.getPoolHits());
        System.out.println("  Pool Misses: " + stats.getPoolMisses());
        System.out.println("  Hit Rate: " + String.format("%.2f", stats.getHitRate() * 100) + "%");
        System.out.println("  CUDA Pool Size: " + stats.getCudaPoolSize());
        System.out.println("  OpenCL Pool Size: " + stats.getOpenclPoolSize());
        
        // Optimize pools
        GPUMemoryPool.optimizePools();
        System.out.println("Memory pools optimized");
    }
    
    /**
     * Optimize batch sizes for different network architectures
     */
    private static void optimizeBatchSizes() {
        System.out.println("--- Batch Size Optimization ---");
        
        // Test different network architectures
        int[][] architectures = {
            {784, 128, 10},   // MNIST-like
            {784, 256, 10},   // Larger hidden layer
            {784, 512, 10},   // Deep network
            {1024, 256, 100}, // Large input/output
            {256, 128, 64}    // Small network
        };
        
        for (int[] arch : architectures) {
            System.out.println("\nOptimizing for architecture: " + arch[0] + " -> " + arch[1] + " -> " + arch[2]);
            
            GPUBatchOptimizer.BatchOptimizationResult result = 
                GPUBatchOptimizer.optimizeBatchSize(arch[0], arch[1], arch[2]);
            
            System.out.println("Best result: " + result);
        }
        
        // Get recommendations
        GPUBatchOptimizer.BatchSizeRecommendations recommendations = GPUBatchOptimizer.getRecommendations();
        if (recommendations != null) {
            System.out.println("\nBatch Size Recommendations:");
            System.out.println("  Optimal: " + recommendations.getOptimalBatchSize());
            System.out.println("  Conservative: " + recommendations.getConservativeBatchSize());
            System.out.println("  Aggressive: " + recommendations.getAggressiveBatchSize());
        }
    }
    
    /**
     * Run comprehensive performance benchmarks
     */
    private static void runPerformanceBenchmarks() {
        System.out.println("--- Performance Benchmarking ---");
        
        // Test different matrix sizes
        int[] matrixSizes = {64, 128, 256, 512, 1024};
        
        for (int size : matrixSizes) {
            System.out.println("\nTesting matrix size: " + size + "x" + size);
            
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
            
            // Test CPU performance
            PerformanceMonitor.reset();
            cpuMatrixMultiply(A, B, C, size, size, size, 1.0, 0.0);
            double cpuTime = PerformanceMonitor.getTotalTime("cpu_matrix_multiply");
            
            double speedup = cpuTime / gpuTime;
            System.out.printf("  GPU: %.3f ms, CPU: %.3f ms, Speedup: %.2fx\n", 
                            gpuTime, cpuTime, speedup);
        }
        
        // Test memory usage
        System.out.println("\nMemory Usage Test:");
        double initialMemory = PerformanceMonitor.getCurrentMemoryMB();
        System.out.println("  Initial Memory: " + initialMemory + " MB");
        
        // Perform intensive operations
        for (int i = 0; i < 50; i++) {
            double[] A = new double[1024];
            double[] B = new double[1024];
            double[] C = new double[1024];
            
            GPUMatrixOps.matrixMultiply(A, B, C, 32, 32, 32, 1.0, 0.0);
            
            if (i % 10 == 0) {
                PerformanceMonitor.updateMemoryUsage();
            }
        }
        
        double peakMemory = PerformanceMonitor.getPeakMemoryMB();
        System.out.println("  Peak Memory: " + peakMemory + " MB");
        System.out.println("  Memory Increase: " + (peakMemory - initialMemory) + " MB");
    }
    
    /**
     * Generate comprehensive optimization report
     */
    private static void generateOptimizationReport() {
        System.out.println("--- Optimization Report ---");
        
        // GPU Information
        System.out.println("\n📋 GPU Information:");
        if (GPUContext.isGPUAvaliable()) {
            GPUContext.GPUInfo gpuInfo = GPUContext.getGPUInfo();
            System.out.println("  GPU Type: " + gpuInfo.getType());
            System.out.println("  GPU Name: " + gpuInfo.getName());
            System.out.println("  GPU Memory: " + (gpuInfo.getGlobalMemory() / (1024 * 1024)) + " MB");
            System.out.println("  Compute Units: " + gpuInfo.getComputeUnits());
            System.out.println("  Max Work Group Size: " + gpuInfo.getMaxWorkGroupSize());
        } else {
            System.out.println("  GPU not available, using CPU fallback");
        }
        
        // Memory Pool Statistics
        System.out.println("\n💾 Memory Pool Statistics:");
        GPUMemoryPool.MemoryPoolStats stats = GPUMemoryPool.getStats();
        System.out.println("  " + stats.toString());
        
        // Performance Summary
        System.out.println("\n⚡ Performance Summary:");
        System.out.println("  GPU Matrix Operations: " + (GPUMatrixOps.isGPUAvaliable() ? "Available" : "Not Available"));
        System.out.println("  Using CUDA: " + (GPUMatrixOps.isUsingCUDA() ? "Yes" : "No"));
        System.out.println("  Memory Pool Hit Rate: " + String.format("%.2f", stats.getHitRate() * 100) + "%");
        
        // Recommendations
        System.out.println("\n🎯 Optimization Recommendations:");
        
        if (stats.getHitRate() < 0.5) {
            System.out.println("  ⚠️  Memory pool hit rate is low. Consider adjusting pool sizes.");
        }
        
        if (stats.getCurrentUsage() > 100 * 1024 * 1024) { // 100MB
            System.out.println("  ⚠️  High memory usage. Consider clearing unused pools.");
        }
        
        if (GPUContext.isGPUAvaliable()) {
            System.out.println("  ✅ GPU acceleration is working properly.");
            System.out.println("  💡 Consider using larger batch sizes for better GPU utilization.");
            System.out.println("  💡 Monitor memory usage during training.");
        } else {
            System.out.println("  ⚠️  GPU not available. Performance will be limited.");
            System.out.println("  💡 Install GPU drivers and CUDA/OpenCL runtime for better performance.");
        }
        
        // Cleanup
        System.out.println("\n🧹 Cleanup:");
        GPUMemoryPool.clearPools();
        System.out.println("  Memory pools cleared");
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
     * Cleanup all GPU resources
     */
    public static void cleanup() {
        System.out.println("\n🧹 Cleaning up GPU resources...");
        
        try {
            GPUMemoryPool.clearPools();
            GPUTestSuite.cleanup();
            System.out.println("✅ GPU resources cleaned up successfully");
        } catch (Exception e) {
            System.err.println("❌ GPU cleanup failed: " + e.getMessage());
        }
    }
} 