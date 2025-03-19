package com.neuralnet.activations;

import com.neuralnet.core.Activation;

public class Tanh implements Activation {
    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derivative(double input) {
        double tanh = activate(input);
        return 1.0 - tanh * tanh;
    }
} 