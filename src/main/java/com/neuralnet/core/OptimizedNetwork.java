package com.neuralnet.core;

import com.neuralnet.activations.Activation;
import com.neuralnet.layers.MatrixLayer;
import com.neuralnet.optimizers.Optimizer;
import com.neuralnet.util.PerformanceMonitor;
import java.util.ArrayList;
import java.util.List;

/**
 * OPTIMIZATION: High-performance network implementation with monitoring
 */
public class OptimizedNetwork {
    private final List<MatrixLayer> layers;
    private final double learningRate;
    private final Loss lossFunction;
    private final Optimizer optimizer;
    private final double[] lastOutput;
    private final boolean enableProfiling;

    public OptimizedNetwork(double learningRate, Loss lossFunction, Optimizer optimizer) {
        this(learningRate, lossFunction, optimizer, false);
    }

    public OptimizedNetwork(double learningRate, Loss lossFunction, Optimizer optimizer, boolean enableProfiling) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.lastOutput = null; // Will be initialized after first layer
        this.enableProfiling = enableProfiling;
    }

    public void addLayer(int inputSize, int outputSize, Activation activation) {
        MatrixLayer layer = new MatrixLayer(inputSize, outputSize, activation, optimizer);
        layers.add(layer);
    }

    public double[] forward(double[] inputs) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }

        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("forward_pass")) {
            // Set inputs to first layer
            layers.get(0).setInputs(inputs);

            // Forward pass through all layers
            for (MatrixLayer layer : layers) {
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

        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("backward_pass")) {
            // Single forward pass, reuse outputs
            double[] predictions = forward(inputs);
            double[] gradients = lossFunction.derivative(predictions, targets);

            // Backward pass through layers
            layers.get(layers.size() - 1).backward(targets);
            for (int i = layers.size() - 2; i >= 0; i--) {
                layers.get(i).backward();
            }

            // Update weights using the optimizer
            for (MatrixLayer layer : layers) {
                layer.updateWeights(learningRate);
            }
        }
    }

    public double computeLoss(double[] predictions, double[] targets) {
        return lossFunction.compute(predictions, targets);
    }

    public List<MatrixLayer> getLayers() {
        return layers;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public Loss getLossFunction() {
        return lossFunction;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * Train the network on a batch of data
     */
    public void trainBatch(double[][] inputs, double[][] targets) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("batch_training")) {
            for (int i = 0; i < inputs.length; i++) {
                backward(inputs[i], targets[i]);
            }
        }
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
} 