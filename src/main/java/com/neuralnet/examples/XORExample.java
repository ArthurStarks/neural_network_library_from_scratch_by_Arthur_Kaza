package com.neuralnet.examples;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import com.neuralnet.training.*;
import com.neuralnet.layers.*;

/**
 * Example demonstrating the XOR problem using our neural network library.
 * This example shows how to:
 * 1. Create a simple feedforward neural network
 * 2. Prepare training data
 * 3. Train the network
 * 4. Make predictions
 */
public class XORExample {
    public static void main(String[] args) {
        // Create training data
        DataSet trainingData = new DataSet(2, 1);
        trainingData.addSample(new double[]{0, 0}, new double[]{0});
        trainingData.addSample(new double[]{0, 1}, new double[]{1});
        trainingData.addSample(new double[]{1, 0}, new double[]{1});
        trainingData.addSample(new double[]{1, 1}, new double[]{0});

        // Create validation data (same as training for this simple example)
        DataSet validationData = new DataSet(2, 1);
        validationData.addSample(new double[]{0, 0}, new double[]{0});
        validationData.addSample(new double[]{0, 1}, new double[]{1});
        validationData.addSample(new double[]{1, 0}, new double[]{1});
        validationData.addSample(new double[]{1, 1}, new double[]{0});

        // Create network
        Network network = new Network();
        
        // Add layers
        network.addLayer(new Layer(2, new ReLU()));     // Input layer
        network.addLayer(new Layer(4, new ReLU()));     // Hidden layer
        network.addLayer(new Layer(1, new Sigmoid()));  // Output layer

        // Create optimizer
        Optimizer optimizer = new Adam(0.9, 0.999, 1e-8);

        // Create trainer
        Trainer trainer = new Trainer(network, trainingData, validationData, 4, 1000);

        // Train the network
        System.out.println("Training network...");
        trainer.train();

        // Test the network
        System.out.println("\nTesting network:");
        testNetwork(network, new double[]{0, 0}, "0 XOR 0");
        testNetwork(network, new double[]{0, 1}, "0 XOR 1");
        testNetwork(network, new double[]{1, 0}, "1 XOR 0");
        testNetwork(network, new double[]{1, 1}, "1 XOR 1");
    }

    private static void testNetwork(Network network, double[] input, String testCase) {
        double[] output = network.forward(input);
        System.out.printf("%s = %.4f%n", testCase, output[0]);
    }
} 