package com.neuralnet.optimizers;

import com.neuralnet.core.Connection;
import com.neuralnet.core.Neuron;

public class Adam implements Optimizer {
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private int t; // Time step

    public Adam(double beta1, double beta2, double epsilon) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
    }

    public Adam() {
        this(0.9, 0.999, 1e-8); // Default values
    }

    @Override
    public void updateWeights(Connection connection, double learningRate) {
        t++;
        double deltaWeight = connection.getDeltaWeight();
        
        // Update biased first moment estimate
        double m = beta1 * connection.getMomentum() + (1 - beta1) * deltaWeight;
        connection.setMomentum(m);
        
        // Update biased second raw moment estimate
        double v = beta2 * connection.getVariance() + (1 - beta2) * deltaWeight * deltaWeight;
        connection.setVariance(v);
        
        // Compute bias-corrected first moment estimate
        double mHat = m / (1 - Math.pow(beta1, t));
        
        // Compute bias-corrected second raw moment estimate
        double vHat = v / (1 - Math.pow(beta2, t));
        
        // Update weights
        connection.setWeight(connection.getWeight() + learningRate * mHat / (Math.sqrt(vHat) + epsilon));
        
        // Reset delta weight
        connection.setDeltaWeight(0.0);
    }

    @Override
    public void updateBias(Neuron neuron, double learningRate) {
        // Similar to weight update but for bias
        double delta = neuron.getDelta();
        double m = beta1 * neuron.getBiasMomentum() + (1 - beta1) * delta;
        double v = beta2 * neuron.getBiasVariance() + (1 - beta2) * delta * delta;
        
        double mHat = m / (1 - Math.pow(beta1, t));
        double vHat = v / (1 - Math.pow(beta2, t));
        
        neuron.setBias(neuron.getBias() + learningRate * mHat / (Math.sqrt(vHat) + epsilon));
        
        // Update momentum and variance for bias
        neuron.setBiasMomentum(m);
        neuron.setBiasVariance(v);
    }

    @Override
    public void initialize(Connection connection) {
        connection.setMomentum(0.0);
        connection.setVariance(0.0);
        connection.setDeltaWeight(0.0);
    }

    @Override
    public void initialize(Neuron neuron) {
        neuron.setBiasMomentum(0.0);
        neuron.setBiasVariance(0.0);
    }
} 