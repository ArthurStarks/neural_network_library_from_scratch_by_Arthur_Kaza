package com.neuralnet.layers;

import com.neuralnet.core.Activation;
import com.neuralnet.core.Layer;
import com.neuralnet.gpu.GPUMatrixOps;
import com.neuralnet.optimizers.Optimizer;
import com.neuralnet.util.PerformanceMonitor;

import java.util.Random;

/**
 * GPU-accelerated layer implementation
 * Uses unified GPU matrix operations for high-performance neural network computations
 */
public class GPULayer extends Layer {
    private final double[][] weights;
    private final double[] biases;
    private final double[] activations;
    private final double[] deltas;
    private final double[] weightedSums;
    private final double[] inputs;
    private final int inputSize;
    private final int outputSize;
    private final Activation activation;
    private static final Random random = new Random();

    public GPULayer(int inputSize, int outputSize, Activation activation, Optimizer optimizer) {
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
        this.inputs = new double[inputSize];
        
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
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("gpu_layer_forward")) {
            // Convert weights to column-major format for GPU
            double[] weightsFlat = new double[inputSize * outputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weightsFlat[i * outputSize + j] = weights[i][j];
                }
            }
            
            // GPU matrix multiplication: weightedSums = inputs * weights + biases
            GPUMatrixOps.matrixMultiply(inputs, weightsFlat, weightedSums, 
                                      1, outputSize, inputSize, 1.0, 0.0);
            
            // Add biases
            GPUMatrixOps.vectorAdd(biases, weightedSums, 1.0);
            
            // Apply activation function
            for (int j = 0; j < outputSize; j++) {
                activations[j] = activation.activate(weightedSums[j]);
            }
        }
    }

    @Override
    public void backward(double[] targets) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("gpu_layer_backward_output")) {
            // Compute output layer deltas
            for (int j = 0; j < outputSize; j++) {
                deltas[j] = (targets[j] - activations[j]) * activation.derivative(activations[j]);
            }
        }
    }

    @Override
    public void backward() {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("gpu_layer_backward_hidden")) {
            // For hidden layers, deltas are computed by the next layer
            // This method is called by the previous layer during backpropagation
            // The deltas are already computed by the next layer
        }
    }

    @Override
    public void updateWeights(double learningRate) {
        try (PerformanceMonitor.Timer timer = new PerformanceMonitor.Timer("gpu_layer_update")) {
            // Convert weights to column-major format for GPU
            double[] weightsFlat = new double[inputSize * outputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weightsFlat[i * outputSize + j] = weights[i][j];
                }
            }
            
            // Compute weight gradients using GPU
            double[] weightGradients = new double[inputSize * outputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weightGradients[i * outputSize + j] = inputs[i] * deltas[j];
                }
            }
            
            // Update weights using GPU
            GPUMatrixOps.vectorAdd(weightGradients, weightsFlat, -learningRate);
            
            // Convert back to row-major format
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights[i][j] = weightsFlat[i * outputSize + j];
                }
            }
            
            // Update biases using GPU
            GPUMatrixOps.vectorAdd(deltas, biases, -learningRate);
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
        // Copy inputs to GPU-optimized buffer
        System.arraycopy(inputs, 0, this.inputs, 0, inputSize);
    }

    /**
     * Get weight gradients for backpropagation to previous layer
     */
    public double[] getWeightGradients() {
        double[] gradients = new double[inputSize];
        
        // Compute gradients for previous layer
        for (int i = 0; i < inputSize; i++) {
            gradients[i] = 0.0;
            for (int j = 0; j < outputSize; j++) {
                gradients[i] += weights[i][j] * deltas[j];
            }
        }
        
        return gradients;
    }

    /**
     * Set deltas from next layer (for hidden layers)
     */
    public void setDeltas(double[] deltas) {
        if (deltas.length != outputSize) {
            throw new IllegalArgumentException("Delta size must match layer output size");
        }
        System.arraycopy(deltas, 0, this.deltas, 0, outputSize);
    }

    /**
     * Get deltas for previous layer
     */
    public double[] getDeltas() {
        return deltas;
    }

    /**
     * Get weighted sums (before activation)
     */
    public double[] getWeightedSums() {
        return weightedSums;
    }

    /**
     * Get weights matrix
     */
    public double[][] getWeights() {
        return weights;
    }

    /**
     * Get biases vector
     */
    public double[] getBiases() {
        return biases;
    }

    /**
     * Get input size
     */
    public int getInputSize() {
        return inputSize;
    }

    /**
     * Get output size
     */
    public int getOutputSize() {
        return outputSize;
    }

    /**
     * Get activation function
     */
    public Activation getActivation() {
        return activation;
    }
} 