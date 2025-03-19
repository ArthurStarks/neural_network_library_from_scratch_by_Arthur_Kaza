package com.neuralnet.core;

import com.neuralnet.optimizers.Optimizer;
import java.util.ArrayList;
import java.util.List;

public class Network {
    private final List<Layer> layers;
    private final double learningRate;
    private final Loss lossFunction;
    private final Optimizer optimizer;

    public Network(double learningRate, Loss lossFunction, Optimizer optimizer) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
    }

    public void addLayer(Layer layer) {
        if (!layers.isEmpty()) {
            layers.get(layers.size() - 1).connectTo(layer);
        }
        layers.add(layer);
    }

    public double[] forward(double[] inputs) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }

        // Set inputs to first layer
        layers.get(0).setInputs(inputs);

        // Forward pass through all layers
        for (Layer layer : layers) {
            layer.forward();
        }

        // Return outputs from last layer
        return layers.get(layers.size() - 1).getOutputs();
    }

    public void backward(double[] inputs, double[] targets) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }

        // Forward pass to compute all outputs
        forward(inputs);

        // Compute loss and gradients
        double[] predictions = layers.get(layers.size() - 1).getOutputs();
        double[] gradients = lossFunction.derivative(predictions, targets);

        // Backward pass through layers
        layers.get(layers.size() - 1).backward(targets);
        for (int i = layers.size() - 2; i >= 0; i--) {
            layers.get(i).backward();
        }

        // Update weights using the optimizer
        for (Layer layer : layers) {
            layer.updateWeights(learningRate);
        }
    }

    public double computeLoss(double[] predictions, double[] targets) {
        return lossFunction.compute(predictions, targets);
    }

    public List<Layer> getLayers() {
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
} 