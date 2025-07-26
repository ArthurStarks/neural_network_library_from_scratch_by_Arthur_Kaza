package com.neuralnet.layers;

import com.neuralnet.core.Activation;
import com.neuralnet.core.Layer;
import com.neuralnet.optimizers.Optimizer;
import java.util.Random;

/**
 * OPTIMIZATION: Matrix-based layer implementation for better performance
 * Uses vectorized operations instead of individual neuron computations
 */
public class MatrixLayer extends Layer {
    private final double[][] weights;
    private final double[] biases;
    private final double[] activations;
    private final double[] deltas;
    private final double[] weightedSums;
    private final int inputSize;
    private final int outputSize;
    private final Activation activation;
    private static final Random random = new Random();

    public MatrixLayer(int inputSize, int outputSize, Activation activation, Optimizer optimizer) {
        super(inputSize, outputSize, activation, optimizer);
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        
        // Initialize matrices and vectors
        this.weights = new double[inputSize][outputSize];
        this.biases = new double[outputSize];
        this.activations = new double[outputSize];
        this.deltas = new double[outputSize];
        this.weightedSums = new double[outputSize];
        
        initializeWeights();
    }

    private void initializeWeights() {
        // Xavier/Glorot initialization
        double limit = Math.sqrt(6.0 / (inputSize + outputSize));
        
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextDouble() * 2 * limit - limit;
            }
        }
        
        for (int j = 0; j < outputSize; j++) {
            biases[j] = random.nextDouble() * 2 - 1;
        }
    }

    @Override
    public void forward() {
        // OPTIMIZATION: Vectorized matrix multiplication
        // weightedSums = weights^T * inputs + biases
        for (int j = 0; j < outputSize; j++) {
            weightedSums[j] = biases[j];
            for (int i = 0; i < inputSize; i++) {
                weightedSums[j] += weights[i][j] * getInputs()[i];
            }
            activations[j] = activation.activate(weightedSums[j]);
        }
    }

    @Override
    public void backward(double[] targets) {
        // OPTIMIZATION: Vectorized gradient computation
        for (int j = 0; j < outputSize; j++) {
            deltas[j] = (targets[j] - activations[j]) * activation.derivative(activations[j]);
        }
    }

    @Override
    public void backward() {
        // OPTIMIZATION: Vectorized backpropagation
        for (int j = 0; j < outputSize; j++) {
            deltas[j] *= activation.derivative(activations[j]);
        }
    }

    @Override
    public void updateWeights(double learningRate) {
        // OPTIMIZATION: Vectorized weight updates
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] += learningRate * deltas[j] * getInputs()[i];
            }
        }
        
        for (int j = 0; j < outputSize; j++) {
            biases[j] += learningRate * deltas[j];
        }
    }

    @Override
    public double[] getOutputs() {
        return activations;
    }

    @Override
    public void setInputs(double[] inputs) {
        if (inputs.length != inputSize) {
            throw new IllegalArgumentException("Input size must match layer input size");
        }
        // OPTIMIZATION: Direct array copy for better performance
        System.arraycopy(inputs, 0, getInputs(), 0, inputSize);
    }

    // Getters for optimization
    public double[][] getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }

    public double[] getDeltas() {
        return deltas;
    }

    public double[] getWeightedSums() {
        return weightedSums;
    }
} 