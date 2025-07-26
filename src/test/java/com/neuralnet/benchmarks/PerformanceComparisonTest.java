package com.neuralnet.benchmarks;

import com.neuralnet.core.Network;
import com.neuralnet.core.OptimizedNetwork;
import com.neuralnet.core.Layer;
import com.neuralnet.activations.ReLU;
import com.neuralnet.activations.Sigmoid;
import com.neuralnet.optimizers.Adam;
import com.neuralnet.loss.MSE;
import com.neuralnet.util.PerformanceMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * Performance comparison between original and optimized network implementations
 */
public class PerformanceComparisonTest {
    private Network originalNetwork;
    private OptimizedNetwork optimizedNetwork;
    private double[][] testInputs;
    private double[][] testTargets;
    private final Random random = new Random(42); // Fixed seed for reproducibility

    @BeforeEach
    public void setup() {
        // Create test data
        int inputSize = 784;
        int hiddenSize = 128;
        int outputSize = 10;
        int batchSize = 100;

        testInputs = new double[batchSize][inputSize];
        testTargets = new double[batchSize][outputSize];

        // Generate test data
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                testInputs[i][j] = random.nextDouble();
            }
            for (int j = 0; j < outputSize; j++) {
                testTargets[i][j] = random.nextDouble();
            }
        }

        // Create original network
        Adam optimizer = new Adam();
        MSE loss = new MSE();
        originalNetwork = new Network(0.01, loss, optimizer);
        originalNetwork.addLayer(new Layer(inputSize, hiddenSize, new ReLU(), optimizer));
        originalNetwork.addLayer(new Layer(hiddenSize, hiddenSize, new ReLU(), optimizer));
        originalNetwork.addLayer(new Layer(hiddenSize, outputSize, new Sigmoid(), optimizer));

        // Create optimized network
        optimizedNetwork = new OptimizedNetwork(0.01, loss, optimizer, true);
        optimizedNetwork.addLayer(inputSize, hiddenSize, new ReLU());
        optimizedNetwork.addLayer(hiddenSize, hiddenSize, new ReLU());
        optimizedNetwork.addLayer(hiddenSize, outputSize, new Sigmoid());
    }

    @Test
    public void testForwardPassPerformance() {
        System.out.println("=== FORWARD PASS PERFORMANCE COMPARISON ===");
        
        // Test original network
        PerformanceMonitor.reset();
        for (int i = 0; i < 1000; i++) {
            originalNetwork.forward(testInputs[i % testInputs.length]);
        }
        double originalForwardTime = PerformanceMonitor.getTotalTime("forward_pass");
        
        // Test optimized network
        PerformanceMonitor.reset();
        for (int i = 0; i < 1000; i++) {
            optimizedNetwork.forward(testInputs[i % testInputs.length]);
        }
        double optimizedForwardTime = PerformanceMonitor.getTotalTime("forward_pass");
        
        System.out.printf("Original Network Forward Pass: %.3f ms\n", originalForwardTime);
        System.out.printf("Optimized Network Forward Pass: %.3f ms\n", optimizedForwardTime);
        System.out.printf("Speedup: %.2fx\n", originalForwardTime / optimizedForwardTime);
        System.out.println();
    }

    @Test
    public void testBackwardPassPerformance() {
        System.out.println("=== BACKWARD PASS PERFORMANCE COMPARISON ===");
        
        // Test original network
        PerformanceMonitor.reset();
        for (int i = 0; i < 1000; i++) {
            originalNetwork.backward(testInputs[i % testInputs.length], testTargets[i % testTargets.length]);
        }
        double originalBackwardTime = PerformanceMonitor.getTotalTime("backward_pass");
        
        // Test optimized network
        PerformanceMonitor.reset();
        for (int i = 0; i < 1000; i++) {
            optimizedNetwork.backward(testInputs[i % testInputs.length], testTargets[i % testTargets.length]);
        }
        double optimizedBackwardTime = PerformanceMonitor.getTotalTime("backward_pass");
        
        System.out.printf("Original Network Backward Pass: %.3f ms\n", originalBackwardTime);
        System.out.printf("Optimized Network Backward Pass: %.3f ms\n", optimizedBackwardTime);
        System.out.printf("Speedup: %.2fx\n", originalBackwardTime / optimizedBackwardTime);
        System.out.println();
    }

    @Test
    public void testBatchTrainingPerformance() {
        System.out.println("=== BATCH TRAINING PERFORMANCE COMPARISON ===");
        
        // Test original network
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 10; epoch++) {
            for (int i = 0; i < testInputs.length; i++) {
                originalNetwork.backward(testInputs[i], testTargets[i]);
            }
        }
        double originalTrainingTime = PerformanceMonitor.getTotalTime("backward_pass");
        
        // Test optimized network
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 10; epoch++) {
            optimizedNetwork.trainBatch(testInputs, testTargets);
        }
        double optimizedTrainingTime = PerformanceMonitor.getTotalTime("batch_training");
        
        System.out.printf("Original Network Training: %.3f ms\n", originalTrainingTime);
        System.out.printf("Optimized Network Training: %.3f ms\n", optimizedTrainingTime);
        System.out.printf("Speedup: %.2fx\n", originalTrainingTime / optimizedTrainingTime);
        System.out.println();
    }

    @Test
    public void testMemoryUsage() {
        System.out.println("=== MEMORY USAGE COMPARISON ===");
        
        // Test original network memory usage
        PerformanceMonitor.reset();
        for (int i = 0; i < 1000; i++) {
            originalNetwork.forward(testInputs[i % testInputs.length]);
            PerformanceMonitor.updateMemoryUsage();
        }
        double originalMemory = PerformanceMonitor.getPeakMemoryMB();
        
        // Test optimized network memory usage
        PerformanceMonitor.reset();
        for (int i = 0; i < 1000; i++) {
            optimizedNetwork.forward(testInputs[i % testInputs.length]);
            PerformanceMonitor.updateMemoryUsage();
        }
        double optimizedMemory = PerformanceMonitor.getPeakMemoryMB();
        
        System.out.printf("Original Network Peak Memory: %.2f MB\n", originalMemory);
        System.out.printf("Optimized Network Peak Memory: %.2f MB\n", optimizedMemory);
        System.out.printf("Memory Reduction: %.1f%%\n", 
                         (originalMemory - optimizedMemory) / originalMemory * 100);
        System.out.println();
    }

    @Test
    public void testEndToEndPerformance() {
        System.out.println("=== END-TO-END PERFORMANCE COMPARISON ===");
        
        // Original network training
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 5; epoch++) {
            for (int i = 0; i < testInputs.length; i++) {
                originalNetwork.forward(testInputs[i]);
                originalNetwork.backward(testInputs[i], testTargets[i]);
            }
        }
        double originalTotalTime = PerformanceMonitor.getTotalTime("forward_pass") + 
                                  PerformanceMonitor.getTotalTime("backward_pass");
        
        // Optimized network training
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 5; epoch++) {
            optimizedNetwork.trainBatch(testInputs, testTargets);
        }
        double optimizedTotalTime = PerformanceMonitor.getTotalTime("batch_training");
        
        System.out.printf("Original Network Total Time: %.3f ms\n", originalTotalTime);
        System.out.printf("Optimized Network Total Time: %.3f ms\n", optimizedTotalTime);
        System.out.printf("Overall Speedup: %.2fx\n", originalTotalTime / optimizedTotalTime);
        
        // Print detailed performance stats
        System.out.println("\nDetailed Performance Statistics:");
        optimizedNetwork.printPerformanceStats();
    }
} 