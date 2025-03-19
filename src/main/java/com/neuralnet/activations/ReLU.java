package com.neuralnet.activations;

import com.neuralnet.core.Activation;

public class ReLU implements Activation {
    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1.0 : 0.0;
    }
} 