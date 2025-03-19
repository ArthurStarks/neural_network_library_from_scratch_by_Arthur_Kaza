package com.neuralnet.layers;

import com.neuralnet.core.Layer;
import com.neuralnet.core.Neuron;
import com.neuralnet.core.Connection;
import com.neuralnet.activations.ActivationFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DropoutLayer extends Layer {
    private final double dropoutRate;
    private final Random random;
    private boolean[] dropoutMask;
    private boolean isTraining;

    public DropoutLayer(int size, double dropoutRate, ActivationFunction activationFunction) {
        super(size, activationFunction);
        this.dropoutRate = dropoutRate;
        this.random = new Random();
        this.dropoutMask = new boolean[size];
        this.isTraining = true;
    }

    @Override
    public void forward(double[] input) {
        if (isTraining) {
            // Generate dropout mask during training
            for (int i = 0; i < dropoutMask.length; i++) {
                dropoutMask[i] = random.nextDouble() > dropoutRate;
            }
        } else {
            // During inference, scale outputs by (1 - dropoutRate)
            for (int i = 0; i < dropoutMask.length; i++) {
                dropoutMask[i] = true;
            }
        }

        // Forward pass with dropout
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            double sum = neuron.getBias();

            for (Connection connection : neuron.getInputConnections()) {
                if (dropoutMask[i]) {
                    sum += connection.getWeight() * connection.getInput().getOutput();
                }
            }

            neuron.setOutput(activationFunction.activate(sum));
        }
    }

    @Override
    public void backward(double[] target) {
        // Backward pass with dropout
        for (int i = 0; i < neurons.size(); i++) {
            if (dropoutMask[i]) {
                Neuron neuron = neurons.get(i);
                double delta = target[i] - neuron.getOutput();
                delta *= activationFunction.derivative(neuron.getOutput());
                neuron.setDelta(delta);

                // Update weights and biases
                for (Connection connection : neuron.getInputConnections()) {
                    connection.setDeltaWeight(connection.getDeltaWeight() + 
                        delta * connection.getInput().getOutput());
                }
                neuron.setDeltaBias(neuron.getDeltaBias() + delta);
            }
        }
    }

    public void setTraining(boolean training) {
        this.isTraining = training;
    }

    public boolean isTraining() {
        return isTraining;
    }

    public double getDropoutRate() {
        return dropoutRate;
    }
} 