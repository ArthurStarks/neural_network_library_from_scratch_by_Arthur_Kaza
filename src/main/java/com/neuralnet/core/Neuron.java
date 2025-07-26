package com.neuralnet.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    private double activationValue;
    private double bias;
    private double delta;
    private double biasMomentum; // For Adam optimizer
    private double biasVariance; // For Adam optimizer
    private final List<Connection> inputConnections;
    private final List<Connection> outputConnections;
    private final Activation activation;
    private static final Random random = new Random();
    
    // OPTIMIZATION: Pre-computed values to avoid recalculation
    private double cachedWeightedSum;
    private boolean weightedSumValid;

    public Neuron(Activation activation) {
        this.activation = activation;
        this.bias = random.nextDouble() * 2 - 1; // Initialize between -1 and 1
        this.inputConnections = new ArrayList<>(8); // OPTIMIZATION: Pre-size for typical connections
        this.outputConnections = new ArrayList<>(8); // OPTIMIZATION: Pre-size for typical connections
        this.delta = 0.0;
        this.biasMomentum = 0.0;
        this.biasVariance = 0.0;
        this.weightedSumValid = false;
    }

    public void computeOutput() {
        if (!weightedSumValid) {
            computeWeightedSum();
        }
        this.activationValue = activation.activate(cachedWeightedSum);
        weightedSumValid = false; // Invalidate cache for next computation
    }
    
    // OPTIMIZATION: Separate weighted sum computation for potential reuse
    private void computeWeightedSum() {
        double weightedSum = bias;
        for (Connection connection : inputConnections) {
            weightedSum += connection.getFromNeuron().getActivationValue() * connection.getWeight();
        }
        this.cachedWeightedSum = weightedSum;
        this.weightedSumValid = true;
    }

    public void computeDelta(double target) {
        this.delta = (target - activationValue) * activation.derivative(activationValue);
        updateConnections();
    }

    public void computeDelta() {
        double weightedSum = 0.0;
        for (Connection connection : outputConnections) {
            weightedSum += connection.getToNeuron().getDelta() * connection.getWeight();
        }
        this.delta = weightedSum * activation.derivative(activationValue);
        updateConnections();
    }

    private void updateConnections() {
        // OPTIMIZATION: Use direct array access for better performance
        for (int i = 0; i < inputConnections.size(); i++) {
            Connection connection = inputConnections.get(i);
            connection.addDeltaWeight(delta * connection.getFromNeuron().getActivationValue());
        }
    }

    public void updateBias(double learningRate) {
        bias += learningRate * delta;
    }

    public void addInputConnection(Connection connection) {
        inputConnections.add(connection);
    }

    public void addOutputConnection(Connection connection) {
        outputConnections.add(connection);
    }

    public double getActivationValue() {
        return activationValue;
    }

    public void setActivationValue(double activationValue) {
        this.activationValue = activationValue;
        this.weightedSumValid = false; // Invalidate cache when input changes
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getDelta() {
        return delta;
    }

    public List<Connection> getInputConnections() {
        return inputConnections;
    }

    public List<Connection> getOutputConnections() {
        return outputConnections;
    }

    public Activation getActivation() {
        return activation;
    }

    public double getBiasMomentum() {
        return biasMomentum;
    }

    public void setBiasMomentum(double biasMomentum) {
        this.biasMomentum = biasMomentum;
    }

    public double getBiasVariance() {
        return biasVariance;
    }

    public void setBiasVariance(double biasVariance) {
        this.biasVariance = biasVariance;
    }
} 