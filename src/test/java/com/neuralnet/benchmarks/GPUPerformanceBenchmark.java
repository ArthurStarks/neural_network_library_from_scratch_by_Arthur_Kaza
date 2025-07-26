package com.neuralnet.benchmarks;

import com.neuralnet.core.Network;
import com.neuralnet.core.OptimizedNetwork;
import com.neuralnet.core.GPUAcceleratedNetwork;
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
 * GPU Performance Benchmark
 * Compares CPU vs GPU performance for neural network operations
 */
public class GPUPerformanceBenchmark {
    private Network cpuNetwork;
    private OptimizedNetwork optimizedNetwork;
    private GPUAcceleratedNetwork gpuNetwork;
    private double[][] testInputs;
    private double[][] testTargets;
    private final Random random = new Random(42); // Fixed seed for reproducibility

    @BeforeEach
    public void setup() {
        // Create test data
        int inputSize = 784;
        int hiddenSize = 256;
        int outputSize = 10;
        int batchSize = 128;

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

        // Create CPU network
        Adam optimizer = new Adam();
        MSE loss = new MSE();
        cpuNetwork = new Network(0.01, loss, optimizer);
        cpuNetwork.addLayer(new Layer(inputSize, hiddenSize, new ReLU(), optimizer));
        cpuNetwork.addLayer(new Layer(hiddenSize, hiddenSize, new ReLU(), optimizer));
        cpuNetwork.addLayer(new Layer(hiddenSize, outputSize, new Sigmoid(), optimizer));

        // Create optimized CPU network
        optimizedNetwork = new OptimizedNetwork(0.01, loss, optimizer, true);
        optimizedNetwork.addLayer(inputSize, hiddenSize, new ReLU());
        optimizedNetwork.addLayer(hiddenSize, hiddenSize, new ReLU());
        optimizedNetwork.addLayer(hiddenSize, outputSize, new Sigmoid());

        // Create GPU network
        gpuNetwork = new GPUAcceleratedNetwork(0.01, loss, optimizer, true);
        gpuNetwork.addLayer(inputSize, hiddenSize, new ReLU());
        gpuNetwork.addLayer(hiddenSize, hiddenSize, new ReLU());
        gpuNetwork.addLayer(hiddenSize, outputSize, new Sigmoid());
    }

    @Test
    public void testGPUInitialization() {
        System.out.println("=== GPU INITIALIZATION TEST ===");
        
        System.out.println("GPU Available: " + gpuNetwork.isGPUAvaliable());
        System.out.println("Using CUDA: " + gpuNetwork.isUsingCUDA());
        System.out.println("GPU Info: " + gpuNetwork.getGPUInfo());
        System.out.println("Network Architecture:");
        System.out.println(gpuNetwork.getArchitectureSummary());
        System.out.println();
    }

    @Test
    public void testForwardPassPerformance() {
        System.out.println("=== FORWARD PASS PERFORMANCE COMPARISON ===");
        
        // Test CPU network
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            cpuNetwork.forward(testInputs[i % testInputs.length]);
        }
        double cpuForwardTime = PerformanceMonitor.getTotalTime("forward_pass");
        
        // Test optimized CPU network
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            optimizedNetwork.forward(testInputs[i % testInputs.length]);
        }
        double optimizedForwardTime = PerformanceMonitor.getTotalTime("forward_pass");
        
        // Test GPU network
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            gpuNetwork.forward(testInputs[i % testInputs.length]);
        }
        double gpuForwardTime = PerformanceMonitor.getTotalTime("gpu_network_forward");
        
        System.out.printf("CPU Network Forward Pass: %.3f ms\n", cpuForwardTime);
        System.out.printf("Optimized CPU Network Forward Pass: %.3f ms\n", optimizedForwardTime);
        System.out.printf("GPU Network Forward Pass: %.3f ms\n", gpuForwardTime);
        System.out.printf("CPU vs GPU Speedup: %.2fx\n", cpuForwardTime / gpuForwardTime);
        System.out.printf("Optimized vs GPU Speedup: %.2fx\n", optimizedForwardTime / gpuForwardTime);
        System.out.println();
    }

    @Test
    public void testBackwardPassPerformance() {
        System.out.println("=== BACKWARD PASS PERFORMANCE COMPARISON ===");
        
        // Test CPU network
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            cpuNetwork.backward(testInputs[i % testInputs.length], testTargets[i % testTargets.length]);
        }
        double cpuBackwardTime = PerformanceMonitor.getTotalTime("backward_pass");
        
        // Test optimized CPU network
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            optimizedNetwork.backward(testInputs[i % testInputs.length], testTargets[i % testTargets.length]);
        }
        double optimizedBackwardTime = PerformanceMonitor.getTotalTime("backward_pass");
        
        // Test GPU network
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            gpuNetwork.backward(testInputs[i % testInputs.length], testTargets[i % testTargets.length]);
        }
        double gpuBackwardTime = PerformanceMonitor.getTotalTime("gpu_network_backward");
        
        System.out.printf("CPU Network Backward Pass: %.3f ms\n", cpuBackwardTime);
        System.out.printf("Optimized CPU Network Backward Pass: %.3f ms\n", optimizedBackwardTime);
        System.out.printf("GPU Network Backward Pass: %.3f ms\n", gpuBackwardTime);
        System.out.printf("CPU vs GPU Speedup: %.2fx\n", cpuBackwardTime / gpuBackwardTime);
        System.out.printf("Optimized vs GPU Speedup: %.2fx\n", optimizedBackwardTime / gpuBackwardTime);
        System.out.println();
    }

    @Test
    public void testBatchTrainingPerformance() {
        System.out.println("=== BATCH TRAINING PERFORMANCE COMPARISON ===");
        
        // Test CPU network
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 5; epoch++) {
            for (int i = 0; i < testInputs.length; i++) {
                cpuNetwork.backward(testInputs[i], testTargets[i]);
            }
        }
        double cpuTrainingTime = PerformanceMonitor.getTotalTime("backward_pass");
        
        // Test optimized CPU network
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 5; epoch++) {
            optimizedNetwork.trainBatch(testInputs, testTargets);
        }
        double optimizedTrainingTime = PerformanceMonitor.getTotalTime("batch_training");
        
        // Test GPU network
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 5; epoch++) {
            gpuNetwork.trainBatch(testInputs, testTargets);
        }
        double gpuTrainingTime = PerformanceMonitor.getTotalTime("gpu_batch_training");
        
        System.out.printf("CPU Network Training: %.3f ms\n", cpuTrainingTime);
        System.out.printf("Optimized CPU Network Training: %.3f ms\n", optimizedTrainingTime);
        System.out.printf("GPU Network Training: %.3f ms\n", gpuTrainingTime);
        System.out.printf("CPU vs GPU Speedup: %.2fx\n", cpuTrainingTime / gpuTrainingTime);
        System.out.printf("Optimized vs GPU Speedup: %.2fx\n", optimizedTrainingTime / gpuTrainingTime);
        System.out.println();
    }

    @Test
    public void testMemoryUsage() {
        System.out.println("=== MEMORY USAGE COMPARISON ===");
        
        // Test CPU network memory usage
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            cpuNetwork.forward(testInputs[i % testInputs.length]);
            PerformanceMonitor.updateMemoryUsage();
        }
        double cpuMemory = PerformanceMonitor.getPeakMemoryMB();
        
        // Test optimized CPU network memory usage
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            optimizedNetwork.forward(testInputs[i % testInputs.length]);
            PerformanceMonitor.updateMemoryUsage();
        }
        double optimizedMemory = PerformanceMonitor.getPeakMemoryMB();
        
        // Test GPU network memory usage
        PerformanceMonitor.reset();
        for (int i = 0; i < 100; i++) {
            gpuNetwork.forward(testInputs[i % testInputs.length]);
            PerformanceMonitor.updateMemoryUsage();
        }
        double gpuMemory = PerformanceMonitor.getPeakMemoryMB();
        
        System.out.printf("CPU Network Peak Memory: %.2f MB\n", cpuMemory);
        System.out.printf("Optimized CPU Network Peak Memory: %.2f MB\n", optimizedMemory);
        System.out.printf("GPU Network Peak Memory: %.2f MB\n", gpuMemory);
        System.out.printf("CPU vs GPU Memory Ratio: %.2fx\n", cpuMemory / gpuMemory);
        System.out.printf("Optimized vs GPU Memory Ratio: %.2fx\n", optimizedMemory / gpuMemory);
        System.out.println();
    }

    @Test
    public void testEndToEndPerformance() {
        System.out.println("=== END-TO-END PERFORMANCE COMPARISON ===");
        
        // CPU network training
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 3; epoch++) {
            for (int i = 0; i < testInputs.length; i++) {
                cpuNetwork.forward(testInputs[i]);
                cpuNetwork.backward(testInputs[i], testTargets[i]);
            }
        }
        double cpuTotalTime = PerformanceMonitor.getTotalTime("forward_pass") + 
                             PerformanceMonitor.getTotalTime("backward_pass");
        
        // Optimized CPU network training
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 3; epoch++) {
            optimizedNetwork.trainBatch(testInputs, testTargets);
        }
        double optimizedTotalTime = PerformanceMonitor.getTotalTime("batch_training");
        
        // GPU network training
        PerformanceMonitor.reset();
        for (int epoch = 0; epoch < 3; epoch++) {
            gpuNetwork.trainBatch(testInputs, testTargets);
        }
        double gpuTotalTime = PerformanceMonitor.getTotalTime("gpu_batch_training");
        
        System.out.printf("CPU Network Total Time: %.3f ms\n", cpuTotalTime);
        System.out.printf("Optimized CPU Network Total Time: %.3f ms\n", optimizedTotalTime);
        System.out.printf("GPU Network Total Time: %.3f ms\n", gpuTotalTime);
        System.out.printf("CPU vs GPU Overall Speedup: %.2fx\n", cpuTotalTime / gpuTotalTime);
        System.out.printf("Optimized vs GPU Overall Speedup: %.2fx\n", optimizedTotalTime / gpuTotalTime);
        
        // Print detailed performance stats
        System.out.println("\nDetailed GPU Performance Statistics:");
        gpuNetwork.printPerformanceStats();
    }

    @Test
    public void testScalability() {
        System.out.println("=== SCALABILITY TEST ===");
        
        // Test with different network sizes
        int[] hiddenSizes = {64, 128, 256, 512, 1024};
        
        for (int hiddenSize : hiddenSizes) {
            System.out.println("Testing with hidden size: " + hiddenSize);
            
            // Create GPU network with current size
            GPUAcceleratedNetwork testNetwork = new GPUAcceleratedNetwork(0.01, new MSE(), new Adam(), true);
            testNetwork.addLayer(784, hiddenSize, new ReLU());
            testNetwork.addLayer(hiddenSize, hiddenSize, new ReLU());
            testNetwork.addLayer(hiddenSize, 10, new Sigmoid());
            
            // Benchmark
            PerformanceMonitor.reset();
            for (int i = 0; i < 50; i++) {
                testNetwork.forward(testInputs[i % testInputs.length]);
            }
            double forwardTime = PerformanceMonitor.getTotalTime("gpu_network_forward");
            
            PerformanceMonitor.reset();
            for (int i = 0; i < 50; i++) {
                testNetwork.backward(testInputs[i % testInputs.length], testTargets[i % testTargets.length]);
            }
            double backwardTime = PerformanceMonitor.getTotalTime("gpu_network_backward");
            
            System.out.printf("  Parameters: %d\n", testNetwork.getParameterCount());
            System.out.printf("  Forward Pass: %.3f ms\n", forwardTime);
            System.out.printf("  Backward Pass: %.3f ms\n", backwardTime);
            System.out.printf("  Total Time: %.3f ms\n", forwardTime + backwardTime);
            System.out.println();
            
            testNetwork.cleanup();
        }
    }
} 