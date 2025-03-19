package com.neuralnet.optimizers;

import com.neuralnet.core.Connection;
import com.neuralnet.core.Neuron;

public class SGD implements Optimizer {
    private final double momentum;
    private final double decay;

    public SGD(double momentum, double decay) {
        this.momentum = momentum;
        this.decay = decay;
    }

    public SGD() {
        this(0.9, 0.0); // Default momentum of 0.9 and no decay
    }

    @Override
    public void updateWeights(Connection connection, double learningRate) {
        // Apply weight decay
        double weightDecay = 1.0 - (learningRate * decay);
        connection.setWeight(connection.getWeight() * weightDecay);
        
        // Update momentum
        double currentMomentum = connection.getMomentum();
        double newMomentum = momentum * currentMomentum + learningRate * connection.getDeltaWeight();
        connection.setMomentum(newMomentum);
        
        // Update weight
        connection.setWeight(connection.getWeight() + newMomentum);
        
        // Reset delta weight
        connection.setDeltaWeight(0.0);
    }

    @Override
    public void updateBias(Neuron neuron, double learningRate) {
        neuron.updateBias(learningRate);
    }

    @Override
    public void initialize(Connection connection) {
        connection.setMomentum(0.0);
        connection.setDeltaWeight(0.0);
    }

    @Override
    public void initialize(Neuron neuron) {
        // No specific initialization needed for SGD
    }
} 