package com.neuralnet.core;

import com.neuralnet.optimizers.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Layer {
    private final List<Neuron> neurons;
    private final int inputSize;
    private final int outputSize;
    private final Optimizer optimizer;
    private static final Random random = new Random();

    public Layer(int inputSize, int outputSize, Activation activation, Optimizer optimizer) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.optimizer = optimizer;
        this.neurons = new ArrayList<>();
        
        for (int i = 0; i < outputSize; i++) {
            neurons.add(new Neuron(activation));
            optimizer.initialize(neurons.get(i));
        }
    }

    public void connectTo(Layer nextLayer) {
        // Xavier/Glorot initialization
        double limit = Math.sqrt(6.0 / (inputSize + outputSize));
        
        for (Neuron currentNeuron : neurons) {
            for (Neuron nextNeuron : nextLayer.getNeurons()) {
                double weight = random.nextDouble() * 2 * limit - limit;
                Connection connection = new Connection(weight, currentNeuron, nextNeuron);
                optimizer.initialize(connection);
                currentNeuron.addOutputConnection(connection);
                nextNeuron.addInputConnection(connection);
            }
        }
    }

    public void forward() {
        for (Neuron neuron : neurons) {
            neuron.computeOutput();
        }
    }

    public void backward(double[] targets) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).computeDelta(targets[i]);
        }
    }

    public void backward() {
        for (Neuron neuron : neurons) {
            neuron.computeDelta();
        }
    }

    public void updateWeights(double learningRate) {
        for (Neuron neuron : neurons) {
            optimizer.updateBias(neuron, learningRate);
            for (Connection connection : neuron.getOutputConnections()) {
                optimizer.updateWeights(connection, learningRate);
            }
        }
    }

    public void setInputs(double[] inputs) {
        if (inputs.length != neurons.size()) {
            throw new IllegalArgumentException("Input size must match layer size");
        }
        for (int i = 0; i < inputs.length; i++) {
            neurons.get(i).setActivationValue(inputs[i]);
        }
    }

    public double[] getOutputs() {
        double[] outputs = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons.get(i).getActivationValue();
        }
        return outputs;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }
} 