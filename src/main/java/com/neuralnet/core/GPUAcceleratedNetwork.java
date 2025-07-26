package com.neuralnet.core;

import com.neuralnet.activations.Activation;
import com.neuralnet.gpu.GPUMatrixOps;
import com.neuralnet.layers.GPULayer;
import com.neuralnet.optimizers.Optimizer;
import com.neuralnet.util.PerformanceMonitor;

import java.util.ArrayList;
import java.util.List;

/**
 * GPU-Accelerated Neural Network
 * Uses GPU layers and matrix operations for maximum performance
 */
public class GPUAcceleratedNetwork {
    private final List<GPULayer> layers;
    private final double learningRate;
    private final Loss lossFunction;
    private final Optimizer optimizer;
    private final boolean enableProfiling;
    private final double[] lastOutput;

    public GPUAcceleratedNetwork(double learningRate, Loss lossFunction, Optimizer optimizer) {
        this(learningRate, lossFunction, optimizer, false);
    }

    public GPUAcceleratedNetwork(double learningRate, Loss lossFunction, Optimizer optimizer, boolean enableProfiling) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.enableProfiling = enableProfiling;
        this.lastOutput = null; // Will be initialized after first layer
        
        // Initialize GPU matrix operations
        GPUMatrixOps.initialize();
    }

    public void addLayer(int inputSize, int outputSize, Activation activation) {
        GPULayer layer = new GPULayer(inputSize, outputSize, activation, optimizer);
        layers.add(layer);
    }

    public double[] forward(double[] inputs) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }

        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("gpu_network_forward")) {
            // Set inputs to first layer
            layers.get(0).setInputs(inputs);

            // Forward pass through all layers
            for (GPULayer layer : layers) {
                layer.forward();
            }

            // Return outputs from last layer
            return layers.get(layers.size() - 1).getOutputs();
        }
    }

    public void backward(double[] inputs, double[] targets) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }

        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("gpu_network_backward")) {
            // Single forward pass, reuse outputs
            double[] predictions = forward(inputs);
            double[] gradients = lossFunction.derivative(predictions, targets);

            // Backward pass through layers
            layers.get(layers.size() - 1).backward(targets);
            
            // Propagate deltas through hidden layers
            for (int i = layers.size() - 2; i >= 0; i--) {
                GPULayer currentLayer = layers.get(i);
                GPULayer nextLayer = layers.get(i + 1);
                
                // Compute deltas for current layer
                double[] nextDeltas = nextLayer.getDeltas();
                double[] currentDeltas = new double[currentLayer.getOutputSize()];
                
                // Propagate deltas: delta_i = sum(delta_j * w_ij) * f'(z_i)
                for (int j = 0; j < currentLayer.getOutputSize(); j++) {
                    double sum = 0.0;
                    for (int k = 0; k < nextLayer.getInputSize(); k++) {
                        sum += nextDeltas[k] * nextLayer.getWeights()[j][k];
                    }
                    currentDeltas[j] = sum * currentLayer.getActivation().derivative(currentLayer.getWeightedSums()[j]);
                }
                
                currentLayer.setDeltas(currentDeltas);
            }

            // Update weights using the optimizer
            for (GPULayer layer : layers) {
                layer.updateWeights(learningRate);
            }
        }
    }

    public double computeLoss(double[] predictions, double[] targets) {
        return lossFunction.compute(predictions, targets);
    }

    /**
     * Train the network on a batch of data
     */
    public void trainBatch(double[][] inputs, double[][] targets) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("gpu_batch_training")) {
            for (int i = 0; i < inputs.length; i++) {
                backward(inputs[i], targets[i]);
            }
        }
    }

    /**
     * Train the network on a single sample
     */
    public void train(double[] inputs, double[] targets) {
        backward(inputs, targets);
    }

    /**
     * Predict output for given inputs
     */
    public double[] predict(double[] inputs) {
        return forward(inputs);
    }

    /**
     * Get network layers
     */
    public List<GPULayer> getLayers() {
        return layers;
    }

    /**
     * Get learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Get loss function
     */
    public Loss getLossFunction() {
        return lossFunction;
    }

    /**
     * Get optimizer
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * Check if GPU acceleration is available
     */
    public boolean isGPUAvaliable() {
        return GPUMatrixOps.isGPUAvaliable();
    }

    /**
     * Check if using CUDA
     */
    public boolean isUsingCUDA() {
        return GPUMatrixOps.isUsingCUDA();
    }

    /**
     * Get GPU information
     */
    public GPUMatrixOps.GPUInfo getGPUInfo() {
        return GPUMatrixOps.getGPUInfo();
    }

    /**
     * Get performance statistics
     */
    public void printPerformanceStats() {
        if (enableProfiling) {
            PerformanceMonitor.printSummary();
        }
    }

    /**
     * Reset performance counters
     */
    public void resetPerformanceCounters() {
        PerformanceMonitor.reset();
    }

    /**
     * Get network parameters count
     */
    public int getParameterCount() {
        int count = 0;
        for (GPULayer layer : layers) {
            count += layer.getInputSize() * layer.getOutputSize(); // weights
            count += layer.getOutputSize(); // biases
        }
        return count;
    }

    /**
     * Get network architecture summary
     */
    public String getArchitectureSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("GPU-Accelerated Network Architecture:\n");
        sb.append("GPU: ").append(getGPUInfo()).append("\n");
        sb.append("Parameters: ").append(getParameterCount()).append("\n");
        sb.append("Layers:\n");
        
        for (int i = 0; i < layers.size(); i++) {
            GPULayer layer = layers.get(i);
            sb.append(String.format("  Layer %d: %d -> %d (%s)\n", 
                                  i, layer.getInputSize(), layer.getOutputSize(), 
                                  layer.getActivation().getClass().getSimpleName()));
        }
        
        return sb.toString();
    }

    /**
     * Cleanup GPU resources
     */
    public void cleanup() {
        GPUMatrixOps.cleanup();
    }
} 