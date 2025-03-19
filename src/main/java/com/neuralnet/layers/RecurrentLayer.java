package com.neuralnet.layers;

import com.neuralnet.core.Layer;
import com.neuralnet.core.Neuron;
import com.neuralnet.core.Connection;
import com.neuralnet.activations.ActivationFunction;
import java.util.ArrayList;
import java.util.List;

public class RecurrentLayer extends Layer {
    private final int hiddenSize;
    private final double[][] weights;
    private final double[][] recurrentWeights;
    private final double[] biases;
    private final double[][] weightGradients;
    private final double[][] recurrentWeightGradients;
    private final double[] biasGradients;
    private double[] previousHiddenState;
    private final List<double[]> hiddenStates;

    public RecurrentLayer(int inputSize, int hiddenSize, ActivationFunction activationFunction) {
        super(hiddenSize, activationFunction);
        
        this.hiddenSize = hiddenSize;
        this.weights = new double[hiddenSize][inputSize];
        this.recurrentWeights = new double[hiddenSize][hiddenSize];
        this.biases = new double[hiddenSize];
        this.weightGradients = new double[hiddenSize][inputSize];
        this.recurrentWeightGradients = new double[hiddenSize][hiddenSize];
        this.biasGradients = new double[hiddenSize];
        this.previousHiddenState = new double[hiddenSize];
        this.hiddenStates = new ArrayList<>();
        
        // Initialize weights with Xavier/Glorot initialization
        double scale = Math.sqrt(2.0 / (inputSize + hiddenSize));
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = (Math.random() * 2 - 1) * scale;
            }
            for (int j = 0; j < hiddenSize; j++) {
                recurrentWeights[i][j] = (Math.random() * 2 - 1) * scale;
            }
            biases[i] = 0.0;
        }
    }

    @Override
    public void forward(double[] input) {
        // Calculate hidden state
        double[] hiddenState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double sum = biases[i];
            
            // Input weights
            for (int j = 0; j < input.length; j++) {
                sum += weights[i][j] * input[j];
            }
            
            // Recurrent weights
            for (int j = 0; j < hiddenSize; j++) {
                sum += recurrentWeights[i][j] * previousHiddenState[j];
            }
            
            hiddenState[i] = activationFunction.activate(sum);
            neurons.get(i).setOutput(hiddenState[i]);
        }
        
        // Store hidden state for next time step
        hiddenStates.add(hiddenState.clone());
        previousHiddenState = hiddenState;
    }

    @Override
    public void backward(double[] target) {
        // Reset gradients
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weightGradients[i][j] = 0.0;
            }
            for (int j = 0; j < hiddenSize; j++) {
                recurrentWeightGradients[i][j] = 0.0;
            }
            biasGradients[i] = 0.0;
        }
        
        // Compute gradients for current time step
        for (int i = 0; i < hiddenSize; i++) {
            Neuron neuron = neurons.get(i);
            double delta = target[i] - neuron.getOutput();
            delta *= activationFunction.derivative(neuron.getOutput());
            
            // Update bias gradient
            biasGradients[i] += delta;
            
            // Update weight gradients
            for (int j = 0; j < weights[0].length; j++) {
                weightGradients[i][j] += delta * input[j];
            }
            
            // Update recurrent weight gradients
            for (int j = 0; j < hiddenSize; j++) {
                recurrentWeightGradients[i][j] += delta * previousHiddenState[j];
            }
        }
        
        // Update weights and biases
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] += weightGradients[i][j];
            }
            for (int j = 0; j < hiddenSize; j++) {
                recurrentWeights[i][j] += recurrentWeightGradients[i][j];
            }
            biases[i] += biasGradients[i];
        }
    }

    public void resetState() {
        previousHiddenState = new double[hiddenSize];
        hiddenStates.clear();
    }

    public double[] getPreviousHiddenState() {
        return previousHiddenState.clone();
    }

    public List<double[]> getHiddenStates() {
        return new ArrayList<>(hiddenStates);
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public double[][] getWeights() {
        return weights.clone();
    }

    public double[][] getRecurrentWeights() {
        return recurrentWeights.clone();
    }

    public double[] getBiases() {
        return biases.clone();
    }
} 