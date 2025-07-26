package com.neuralnet.gpu;

import com.neuralnet.util.PerformanceMonitor;
import java.util.*;
import java.util.logging.Logger;

/**
 * GPU Batch Size Optimizer
 * Finds optimal batch sizes for maximum GPU utilization and performance
 */
public class GPUBatchOptimizer {
    private static final Logger logger = Logger.getLogger(GPUBatchOptimizer.class.getName());
    
    // Optimization results
    private static BatchOptimizationResult bestResult;
    private static Map<Integer, BatchOptimizationResult> batchResults = new HashMap<>();
    
    /**
     * Optimize batch size for given network architecture
     */
    public static BatchOptimizationResult optimizeBatchSize(int inputSize, int hiddenSize, int outputSize) {
        System.out.println("=== GPU BATCH SIZE OPTIMIZATION ===");
        System.out.println("Network: " + inputSize + " -> " + hiddenSize + " -> " + outputSize);
        
        // Test different batch sizes
        int[] batchSizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
        List<BatchOptimizationResult> results = new ArrayList<>();
        
        for (int batchSize : batchSizes) {
            BatchOptimizationResult result = testBatchSize(inputSize, hiddenSize, outputSize, batchSize);
            results.add(result);
            batchResults.put(batchSize, result);
            
            System.out.printf("Batch Size %d: %.3fms, %.2f samples/sec, %.2f MB\n", 
                            batchSize, result.getAverageTime(), result.getThroughput(), result.getMemoryUsage());
        }
        
        // Find optimal batch size
        bestResult = findOptimalBatchSize(results);
        
        System.out.println("\n=== OPTIMIZATION RESULTS ===");
        System.out.println("Optimal Batch Size: " + bestResult.getBatchSize());
        System.out.println("Best Throughput: " + String.format("%.2f", bestResult.getThroughput()) + " samples/sec");
        System.out.println("Best Time: " + String.format("%.3f", bestResult.getAverageTime()) + " ms");
        System.out.println("Memory Usage: " + String.format("%.2f", bestResult.getMemoryUsage()) + " MB");
        
        return bestResult;
    }
    
    /**
     * Test specific batch size
     */
    private static BatchOptimizationResult testBatchSize(int inputSize, int hiddenSize, int outputSize, int batchSize) {
        // Create test data
        double[][] inputs = new double[batchSize][inputSize];
        double[][] targets = new double[batchSize][outputSize];
        
        // Initialize with random data
        Random random = new Random(42);
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                inputs[i][j] = random.nextDouble();
            }
            for (int j = 0; j < outputSize; j++) {
                targets[i][j] = random.nextDouble();
            }
        }
        
        // Test forward pass
        PerformanceMonitor.reset();
        double forwardTime = testForwardPass(inputs, targets);
        
        // Test backward pass
        PerformanceMonitor.reset();
        double backwardTime = testBackwardPass(inputs, targets);
        
        // Test full training step
        PerformanceMonitor.reset();
        double trainingTime = testTrainingStep(inputs, targets);
        
        // Calculate metrics
        double totalTime = forwardTime + backwardTime;
        double throughput = (batchSize * 1000.0) / totalTime; // samples per second
        double memoryUsage = PerformanceMonitor.getPeakMemoryMB();
        
        return new BatchOptimizationResult(
            batchSize, totalTime, throughput, memoryUsage, forwardTime, backwardTime, trainingTime
        );
    }
    
    /**
     * Test forward pass performance
     */
    private static double testForwardPass(double[][] inputs, double[][] targets) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("forward_pass")) {
            // Simulate forward pass with GPU matrix operations
            for (int i = 0; i < inputs.length; i++) {
                // Matrix multiplication: input -> hidden
                double[] hidden = new double[inputs[i].length];
                GPUMatrixOps.matrixMultiply(inputs[i], new double[inputs[i].length * hidden.length], 
                                          hidden, 1, hidden.length, inputs[i].length, 1.0, 0.0);
                
                // Matrix multiplication: hidden -> output
                double[] output = new double[targets[i].length];
                GPUMatrixOps.matrixMultiply(hidden, new double[hidden.length * output.length], 
                                          output, 1, output.length, hidden.length, 1.0, 0.0);
            }
        }
        return PerformanceMonitor.getTotalTime("forward_pass");
    }
    
    /**
     * Test backward pass performance
     */
    private static double testBackwardPass(double[][] inputs, double[][] targets) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("backward_pass")) {
            // Simulate backward pass with GPU matrix operations
            for (int i = 0; i < inputs.length; i++) {
                // Gradient computation
                double[] gradients = new double[targets[i].length];
                GPUMatrixOps.elementWiseMultiply(targets[i], gradients);
                
                // Weight updates
                double[] weightGradients = new double[inputs[i].length * gradients.length];
                GPUMatrixOps.matrixMultiply(inputs[i], gradients, weightGradients, 
                                          inputs[i].length, gradients.length, 1, 1.0, 0.0);
            }
        }
        return PerformanceMonitor.getTotalTime("backward_pass");
    }
    
    /**
     * Test full training step performance
     */
    private static double testTrainingStep(double[][] inputs, double[][] targets) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("training_step")) {
            // Simulate complete training step
            for (int i = 0; i < inputs.length; i++) {
                // Forward pass
                double[] hidden = new double[inputs[i].length];
                GPUMatrixOps.matrixMultiply(inputs[i], new double[inputs[i].length * hidden.length], 
                                          hidden, 1, hidden.length, inputs[i].length, 1.0, 0.0);
                
                double[] output = new double[targets[i].length];
                GPUMatrixOps.matrixMultiply(hidden, new double[hidden.length * output.length], 
                                          output, 1, output.length, hidden.length, 1.0, 0.0);
                
                // Backward pass
                double[] gradients = new double[targets[i].length];
                GPUMatrixOps.elementWiseMultiply(targets[i], gradients);
                
                double[] weightGradients = new double[inputs[i].length * gradients.length];
                GPUMatrixOps.matrixMultiply(inputs[i], gradients, weightGradients, 
                                          inputs[i].length, gradients.length, 1, 1.0, 0.0);
            }
        }
        return PerformanceMonitor.getTotalTime("training_step");
    }
    
    /**
     * Find optimal batch size based on multiple criteria
     */
    private static BatchOptimizationResult findOptimalBatchSize(List<BatchOptimizationResult> results) {
        BatchOptimizationResult bestResult = null;
        double bestScore = -1;
        
        for (BatchOptimizationResult result : results) {
            // Calculate composite score based on throughput, memory efficiency, and time
            double throughputScore = result.getThroughput() / 1000.0; // Normalize to reasonable range
            double memoryScore = 1.0 / (result.getMemoryUsage() / 100.0); // Lower memory is better
            double timeScore = 1.0 / (result.getAverageTime() / 100.0); // Lower time is better
            
            // Weighted composite score
            double score = throughputScore * 0.5 + memoryScore * 0.3 + timeScore * 0.2;
            
            if (score > bestScore) {
                bestScore = score;
                bestResult = result;
            }
        }
        
        return bestResult;
    }
    
    /**
     * Get recommended batch sizes for different scenarios
     */
    public static BatchSizeRecommendations getRecommendations() {
        if (bestResult == null) {
            return null;
        }
        
        return new BatchSizeRecommendations(
            bestResult.getBatchSize(),
            bestResult.getBatchSize() / 2, // Conservative
            bestResult.getBatchSize() * 2,  // Aggressive
            batchResults
        );
    }
    
    /**
     * Batch optimization result
     */
    public static class BatchOptimizationResult {
        private final int batchSize;
        private final double averageTime;
        private final double throughput;
        private final double memoryUsage;
        private final double forwardTime;
        private final double backwardTime;
        private final double trainingTime;
        
        public BatchOptimizationResult(int batchSize, double averageTime, double throughput, 
                                    double memoryUsage, double forwardTime, double backwardTime, double trainingTime) {
            this.batchSize = batchSize;
            this.averageTime = averageTime;
            this.throughput = throughput;
            this.memoryUsage = memoryUsage;
            this.forwardTime = forwardTime;
            this.backwardTime = backwardTime;
            this.trainingTime = trainingTime;
        }
        
        public int getBatchSize() { return batchSize; }
        public double getAverageTime() { return averageTime; }
        public double getThroughput() { return throughput; }
        public double getMemoryUsage() { return memoryUsage; }
        public double getForwardTime() { return forwardTime; }
        public double getBackwardTime() { return backwardTime; }
        public double getTrainingTime() { return trainingTime; }
        
        @Override
        public String toString() {
            return String.format("BatchSize=%d, Time=%.3fms, Throughput=%.2f/s, Memory=%.2fMB", 
                               batchSize, averageTime, throughput, memoryUsage);
        }
    }
    
    /**
     * Batch size recommendations
     */
    public static class BatchSizeRecommendations {
        private final int optimalBatchSize;
        private final int conservativeBatchSize;
        private final int aggressiveBatchSize;
        private final Map<Integer, BatchOptimizationResult> allResults;
        
        public BatchSizeRecommendations(int optimalBatchSize, int conservativeBatchSize, 
                                     int aggressiveBatchSize, Map<Integer, BatchOptimizationResult> allResults) {
            this.optimalBatchSize = optimalBatchSize;
            this.conservativeBatchSize = conservativeBatchSize;
            this.aggressiveBatchSize = aggressiveBatchSize;
            this.allResults = allResults;
        }
        
        public int getOptimalBatchSize() { return optimalBatchSize; }
        public int getConservativeBatchSize() { return conservativeBatchSize; }
        public int getAggressiveBatchSize() { return aggressiveBatchSize; }
        public Map<Integer, BatchOptimizationResult> getAllResults() { return allResults; }
        
        @Override
        public String toString() {
            return String.format("BatchSizeRecommendations{optimal=%d, conservative=%d, aggressive=%d}", 
                               optimalBatchSize, conservativeBatchSize, aggressiveBatchSize);
        }
    }
} 