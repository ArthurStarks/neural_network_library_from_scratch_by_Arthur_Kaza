package com.neuralnet.activations;

import com.neuralnet.core.Activation;

public class Softmax implements Activation {
    @Override
    public double activate(double input) {
        // Note: This method is not used for Softmax as it requires the entire array of inputs
        throw new UnsupportedOperationException("Softmax requires the entire array of inputs");
    }

    @Override
    public double derivative(double input) {
        // Note: This method is not used for Softmax as it requires the entire array of inputs
        throw new UnsupportedOperationException("Softmax requires the entire array of inputs");
    }

    public double[] activate(double[] inputs) {
        double[] outputs = new double[inputs.length];
        double maxInput = inputs[0];
        
        // Find the maximum input to prevent numerical instability
        for (double input : inputs) {
            maxInput = Math.max(maxInput, input);
        }
        
        // Compute the sum of exp(inputs - maxInput)
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = Math.exp(inputs[i] - maxInput);
            sum += outputs[i];
        }
        
        // Normalize
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] /= sum;
        }
        
        return outputs;
    }

    public double[] derivative(double[] inputs, int targetIndex) {
        double[] outputs = activate(inputs);
        double[] derivatives = new double[outputs.length];
        
        for (int i = 0; i < outputs.length; i++) {
            if (i == targetIndex) {
                derivatives[i] = outputs[i] * (1 - outputs[i]);
            } else {
                derivatives[i] = -outputs[i] * outputs[targetIndex];
            }
        }
        
        return derivatives;
    }
} 