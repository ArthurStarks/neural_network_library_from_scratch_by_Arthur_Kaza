package com.neuralnet.activations;

import com.neuralnet.core.Activation;

public class Sigmoid implements Activation {
    @Override
    public double activate(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
        double sigmoid = activate(input);
        return sigmoid * (1.0 - sigmoid);
    }
} 