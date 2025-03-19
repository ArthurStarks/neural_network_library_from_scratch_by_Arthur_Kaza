package com.neuralnet.layers;

import com.neuralnet.core.Layer;
import com.neuralnet.core.Neuron;
import com.neuralnet.core.Connection;
import com.neuralnet.activations.ActivationFunction;
import java.util.ArrayList;
import java.util.List;

public class BatchNormalizationLayer extends Layer {
    private final double epsilon;
    private final double momentum;
    private final double[] gamma;
    private final double[] beta;
    private final double[] runningMean;
    private final double[] runningVariance;
    private boolean isTraining;
    private double[] currentBatchMean;
    private double[] currentBatchVariance;

    public BatchNormalizationLayer(int size, double epsilon, double momentum, 
                                 ActivationFunction activationFunction) {
        super(size, activationFunction);
        this.epsilon = epsilon;
        this.momentum = momentum;
        this.gamma = new double[size];
        this.beta = new double[size];
        this.runningMean = new double[size];
        this.runningVariance = new double[size];
        this.isTraining = true;
        
        // Initialize gamma and beta
        for (int i = 0; i < size; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    @Override
    public void forward(double[] input) {
        if (isTraining) {
            // Calculate batch statistics
            currentBatchMean = new double[neurons.size()];
            currentBatchVariance = new double[neurons.size()];
            
            // First pass: calculate mean
            for (int i = 0; i < neurons.size(); i++) {
                Neuron neuron = neurons.get(i);
                double sum = neuron.getBias();
                for (Connection connection : neuron.getInputConnections()) {
                    sum += connection.getWeight() * connection.getInput().getOutput();
                }
                currentBatchMean[i] = sum;
            }
            
            // Second pass: calculate variance
            for (int i = 0; i < neurons.size(); i++) {
                double sumSquaredDiff = 0.0;
                for (int j = 0; j < neurons.size(); j++) {
                    double diff = currentBatchMean[j] - currentBatchMean[i];
                    sumSquaredDiff += diff * diff;
                }
                currentBatchVariance[i] = sumSquaredDiff / neurons.size();
                
                // Update running statistics
                runningMean[i] = momentum * runningMean[i] + 
                               (1 - momentum) * currentBatchMean[i];
                runningVariance[i] = momentum * runningVariance[i] + 
                                   (1 - momentum) * currentBatchVariance[i];
            }
        }

        // Normalize and scale
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            double sum = neuron.getBias();
            for (Connection connection : neuron.getInputConnections()) {
                sum += connection.getWeight() * connection.getInput().getOutput();
            }

            double mean = isTraining ? currentBatchMean[i] : runningMean[i];
            double variance = isTraining ? currentBatchVariance[i] : runningVariance[i];
            double normalized = (sum - mean) / Math.sqrt(variance + epsilon);
            double scaled = gamma[i] * normalized + beta[i];

            neuron.setOutput(activationFunction.activate(scaled));
        }
    }

    @Override
    public void backward(double[] target) {
        // Calculate gradients for gamma and beta
        double[] dGamma = new double[neurons.size()];
        double[] dBeta = new double[neurons.size()];
        
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            double delta = target[i] - neuron.getOutput();
            delta *= activationFunction.derivative(neuron.getOutput());
            
            // Update gamma and beta gradients
            double normalized = (neuron.getOutput() - currentBatchMean[i]) / 
                              Math.sqrt(currentBatchVariance[i] + epsilon);
            dGamma[i] = delta * normalized;
            dBeta[i] = delta;
            
            // Update weights and biases
            for (Connection connection : neuron.getInputConnections()) {
                connection.setDeltaWeight(connection.getDeltaWeight() + 
                    delta * connection.getInput().getOutput());
            }
            neuron.setDeltaBias(neuron.getDeltaBias() + delta);
        }
        
        // Update gamma and beta
        for (int i = 0; i < neurons.size(); i++) {
            gamma[i] += dGamma[i];
            beta[i] += dBeta[i];
        }
    }

    public void setTraining(boolean training) {
        this.isTraining = training;
    }

    public boolean isTraining() {
        return isTraining;
    }

    public double[] getGamma() {
        return gamma.clone();
    }

    public double[] getBeta() {
        return beta.clone();
    }

    public double[] getRunningMean() {
        return runningMean.clone();
    }

    public double[] getRunningVariance() {
        return runningVariance.clone();
    }
} 