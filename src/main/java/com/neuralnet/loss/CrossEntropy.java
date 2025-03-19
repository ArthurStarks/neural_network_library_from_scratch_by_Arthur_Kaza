package com.neuralnet.loss;

public class CrossEntropy implements Loss {
    private static final double EPSILON = 1e-15; // Small constant to prevent log(0)

    @Override
    public double compute(double[] predictions, double[] targets) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same length");
        }

        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            // Clip predictions to prevent log(0)
            double prediction = Math.max(EPSILON, Math.min(1 - EPSILON, predictions[i]));
            sum += targets[i] * Math.log(prediction);
        }
        return -sum;
    }

    @Override
    public double[] derivative(double[] predictions, double[] targets) {
        if (predictions.length != targets.length) {
            throw new IllegalArgumentException("Predictions and targets must have the same length");
        }

        double[] derivatives = new double[predictions.length];
        for (int i = 0; i < predictions.length; i++) {
            // Clip predictions to prevent division by zero
            double prediction = Math.max(EPSILON, Math.min(1 - EPSILON, predictions[i]));
            derivatives[i] = -targets[i] / prediction;
        }
        return derivatives;
    }

    // Binary cross-entropy for single output
    public double computeBinary(double prediction, double target) {
        prediction = Math.max(EPSILON, Math.min(1 - EPSILON, prediction));
        return -(target * Math.log(prediction) + (1 - target) * Math.log(1 - prediction));
    }

    public double derivativeBinary(double prediction, double target) {
        prediction = Math.max(EPSILON, Math.min(1 - EPSILON, prediction));
        return -(target / prediction - (1 - target) / (1 - prediction));
    }
} 