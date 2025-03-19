package com.neuralnet.examples;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import com.neuralnet.training.*;
import com.neuralnet.layers.*;
import java.util.Random;

/**
 * Example demonstrating time series prediction using our neural network library.
 * This example shows how to:
 * 1. Generate synthetic time series data
 * 2. Create an RNN architecture
 * 3. Train the network
 * 4. Make predictions
 */
public class TimeSeriesExample {
    private static final int SEQUENCE_LENGTH = 50;
    private static final int HIDDEN_SIZE = 32;
    private static final int BATCH_SIZE = 16;
    private static final int EPOCHS = 100;

    public static void main(String[] args) {
        // Generate synthetic time series data
        System.out.println("Generating time series data...");
        DataSet trainingData = generateTimeSeriesData(1000);
        DataSet validationData = generateTimeSeriesData(200);

        // Create network
        Network network = new Network();
        
        // Add layers
        network.addLayer(new RecurrentLayer(1, HIDDEN_SIZE, new Tanh()));
        network.addLayer(new Layer(HIDDEN_SIZE, new ReLU()));
        network.addLayer(new Layer(1, new Linear()));

        // Create optimizer
        Optimizer optimizer = new Adam(0.9, 0.999, 1e-8);

        // Create trainer
        Trainer trainer = new Trainer(network, trainingData, validationData, BATCH_SIZE, EPOCHS);

        // Train the network
        System.out.println("Training network...");
        trainer.train();

        // Test the network
        System.out.println("\nTesting network:");
        testNetwork(network, generateTimeSeriesData(1));
    }

    private static DataSet generateTimeSeriesData(int numSamples) {
        DataSet dataset = new DataSet(SEQUENCE_LENGTH, 1);
        Random random = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < numSamples; i++) {
            // Generate input sequence
            double[] input = new double[SEQUENCE_LENGTH];
            double[] target = new double[1];
            
            // Generate a sine wave with random phase and frequency
            double phase = random.nextDouble() * 2 * Math.PI;
            double frequency = 0.1 + random.nextDouble() * 0.2;
            
            for (int t = 0; t < SEQUENCE_LENGTH; t++) {
                input[t] = Math.sin(frequency * t + phase);
            }
            
            // Target is the next value in the sequence
            target[0] = Math.sin(frequency * SEQUENCE_LENGTH + phase);
            
            dataset.addSample(input, target);
        }
        
        return dataset;
    }

    private static void testNetwork(Network network, DataSet testData) {
        double[] input = testData.getInputs().get(0);
        double[] target = testData.getTargets().get(0);
        
        // Print first few values of the input sequence
        System.out.println("Input sequence (first 10 values):");
        for (int i = 0; i < 10; i++) {
            System.out.printf("%.4f ", input[i]);
        }
        System.out.println();
        
        // Make prediction
        double[] output = network.forward(input);
        
        System.out.printf("Target value: %.4f%n", target[0]);
        System.out.printf("Predicted value: %.4f%n", output[0]);
        System.out.printf("Error: %.4f%n", Math.abs(target[0] - output[0]));
    }
} 