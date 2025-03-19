package com.neuralnet.core;

public class Connection {
    private double weight;
    private final Neuron fromNeuron;
    private final Neuron toNeuron;
    private double deltaWeight;
    private double momentum;
    private double variance; // For Adam optimizer
    private static final double MOMENTUM_FACTOR = 0.9;

    public Connection(double weight, Neuron fromNeuron, Neuron toNeuron) {
        this.weight = weight;
        this.fromNeuron = fromNeuron;
        this.toNeuron = toNeuron;
        this.deltaWeight = 0.0;
        this.momentum = 0.0;
        this.variance = 0.0;
    }

    public void updateWeight(double learningRate) {
        // Update momentum
        momentum = MOMENTUM_FACTOR * momentum + learningRate * deltaWeight;
        
        // Update weight with momentum
        weight += momentum;
        
        // Reset delta weight
        deltaWeight = 0.0;
    }

    public void addDeltaWeight(double delta) {
        this.deltaWeight += delta;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public Neuron getFromNeuron() {
        return fromNeuron;
    }

    public Neuron getToNeuron() {
        return toNeuron;
    }

    public double getDeltaWeight() {
        return deltaWeight;
    }

    public void setDeltaWeight(double deltaWeight) {
        this.deltaWeight = deltaWeight;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public double getVariance() {
        return variance;
    }

    public void setVariance(double variance) {
        this.variance = variance;
    }
} 