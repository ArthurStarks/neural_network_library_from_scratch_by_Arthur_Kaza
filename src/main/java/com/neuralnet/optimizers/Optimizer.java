package com.neuralnet.optimizers;

import com.neuralnet.core.Connection;
import com.neuralnet.core.Neuron;

public interface Optimizer {
    void updateWeights(Connection connection, double learningRate);
    void updateBias(Neuron neuron, double learningRate);
    void initialize(Connection connection);
    void initialize(Neuron neuron);
} 