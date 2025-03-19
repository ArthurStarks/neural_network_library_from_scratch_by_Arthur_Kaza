package com.neuralnet.layers;

import com.neuralnet.core.Layer;
import com.neuralnet.core.Neuron;
import com.neuralnet.core.Connection;
import com.neuralnet.activations.ActivationFunction;
import java.util.ArrayList;
import java.util.List;

public class ConvolutionalLayer extends Layer {
    private final int inputWidth;
    private final int inputHeight;
    private final int inputChannels;
    private final int kernelSize;
    private final int stride;
    private final int numFilters;
    private final double[][][][] kernels;
    private final double[] biases;
    private final double[][][][] kernelGradients;
    private final double[] biasGradients;

    public ConvolutionalLayer(int inputWidth, int inputHeight, int inputChannels,
                            int kernelSize, int stride, int numFilters,
                            ActivationFunction activationFunction) {
        super(calculateOutputSize(inputWidth, inputHeight, kernelSize, stride) * numFilters,
              activationFunction);
        
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputChannels = inputChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.numFilters = numFilters;
        
        // Initialize kernels and biases
        this.kernels = new double[numFilters][inputChannels][kernelSize][kernelSize];
        this.biases = new double[numFilters];
        this.kernelGradients = new double[numFilters][inputChannels][kernelSize][kernelSize];
        this.biasGradients = new double[numFilters];
        
        // Initialize weights with Xavier/Glorot initialization
        double scale = Math.sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        for (int f = 0; f < numFilters; f++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        kernels[f][c][i][j] = (Math.random() * 2 - 1) * scale;
                    }
                }
            }
            biases[f] = 0.0;
        }
    }

    private static int calculateOutputSize(int inputSize, int kernelSize, int stride) {
        return (inputSize - kernelSize) / stride + 1;
    }

    @Override
    public void forward(double[] input) {
        // Reshape input to 3D array (channels, height, width)
        double[][][] input3D = reshapeInput(input);
        
        // Calculate output dimensions
        int outputWidth = calculateOutputSize(inputWidth, kernelSize, stride);
        int outputHeight = calculateOutputSize(inputHeight, kernelSize, stride);
        
        // Perform convolution for each filter
        for (int f = 0; f < numFilters; f++) {
            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    double sum = biases[f];
                    
                    // Convolve with each input channel
                    for (int c = 0; c < inputChannels; c++) {
                        for (int ky = 0; ky < kernelSize; ky++) {
                            for (int kx = 0; kx < kernelSize; kx++) {
                                int inputY = y * stride + ky;
                                int inputX = x * stride + kx;
                                sum += input3D[c][inputY][inputX] * kernels[f][c][ky][kx];
                            }
                        }
                    }
                    
                    // Store output
                    int outputIndex = f * outputWidth * outputHeight + y * outputWidth + x;
                    neurons.get(outputIndex).setOutput(activationFunction.activate(sum));
                }
            }
        }
    }

    @Override
    public void backward(double[] target) {
        // Calculate output dimensions
        int outputWidth = calculateOutputSize(inputWidth, kernelSize, stride);
        int outputHeight = calculateOutputSize(inputHeight, kernelSize, stride);
        
        // Reset gradients
        for (int f = 0; f < numFilters; f++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        kernelGradients[f][c][i][j] = 0.0;
                    }
                }
            }
            biasGradients[f] = 0.0;
        }
        
        // Compute gradients
        for (int f = 0; f < numFilters; f++) {
            for (int y = 0; y < outputHeight; y++) {
                for (int x = 0; x < outputWidth; x++) {
                    int outputIndex = f * outputWidth * outputHeight + y * outputWidth + x;
                    Neuron neuron = neurons.get(outputIndex);
                    double delta = target[outputIndex] - neuron.getOutput();
                    delta *= activationFunction.derivative(neuron.getOutput());
                    
                    // Update bias gradient
                    biasGradients[f] += delta;
                    
                    // Update kernel gradients
                    for (int c = 0; c < inputChannels; c++) {
                        for (int ky = 0; ky < kernelSize; ky++) {
                            for (int kx = 0; kx < kernelSize; kx++) {
                                int inputY = y * stride + ky;
                                int inputX = x * stride + kx;
                                kernelGradients[f][c][ky][kx] += delta * 
                                    input3D[c][inputY][inputX];
                            }
                        }
                    }
                }
            }
        }
        
        // Update kernels and biases
        for (int f = 0; f < numFilters; f++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        kernels[f][c][i][j] += kernelGradients[f][c][i][j];
                    }
                }
            }
            biases[f] += biasGradients[f];
        }
    }

    private double[][][] reshapeInput(double[] input) {
        double[][][] input3D = new double[inputChannels][inputHeight][inputWidth];
        int index = 0;
        for (int c = 0; c < inputChannels; c++) {
            for (int y = 0; y < inputHeight; y++) {
                for (int x = 0; x < inputWidth; x++) {
                    input3D[c][y][x] = input[index++];
                }
            }
        }
        return input3D;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    public int getInputChannels() {
        return inputChannels;
    }

    public int getKernelSize() {
        return kernelSize;
    }

    public int getStride() {
        return stride;
    }

    public int getNumFilters() {
        return numFilters;
    }

    public double[][][][] getKernels() {
        return kernels.clone();
    }

    public double[] getBiases() {
        return biases.clone();
    }
} 