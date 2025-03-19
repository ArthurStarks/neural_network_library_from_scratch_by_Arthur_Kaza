package com.neuralnet.loss;

public class MSE implements Loss {
    @Override
    public double compute(double[] predictions, double[] targets) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same length");
        }

        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        return sum / predictions.length;
    }

    @Override
    public double[] derivative(double[] predictions, double[] targets) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same length");
        }

        double[] derivatives = new double[predictions.length];
        for (int i = 0; i < predictions.length; i++) {
            // For MSE, the derivative is 2 * (prediction - target) / n
            derivatives[i] = 2.0 * (predictions[i] - targets[i]) / predictions.length;
        }
        return derivatives;
    }
} 