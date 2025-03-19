package com.neuralnet.loss;

public interface Loss {
    double compute(double[] predictions, double[] targets);
    double[] derivative(double[] predictions, double[] targets);
} 