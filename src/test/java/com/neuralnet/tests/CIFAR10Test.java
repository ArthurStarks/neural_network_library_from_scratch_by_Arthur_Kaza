package com.neuralnet.tests;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import com.neuralnet.training.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.util.Random;

public class CIFAR10Test {
    private static final int IMAGE_SIZE = 32;
    private static final int NUM_CHANNELS = 3;
    private static final int NUM_CLASSES = 10;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 5;
    private static final double MIN_ACCURACY = 0.45; // Minimum expected accuracy on CIFAR-10

    private Network network;
    private DataSet trainingData;
    private DataSet validationData;
    private Optimizer optimizer;
    private Trainer trainer;

    @BeforeEach
    void setUp() {
        // Create network
        optimizer = new Adam();
        network = new Network(optimizer);
        
        // Add layers (CNN-like architecture)
        network.addLayer(new Layer(IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS, new ReLU(), optimizer));  // Input layer
        network.addLayer(new Layer(512, new ReLU(), optimizer));                                     // Hidden layer
        network.addLayer(new Layer(256, new ReLU(), optimizer));                                     // Hidden layer
        network.addLayer(new Layer(128, new ReLU(), optimizer));                                     // Hidden layer
        network.addLayer(new Layer(NUM_CLASSES, new Softmax(), optimizer));                         // Output layer

        // Load CIFAR-10 data
        try {
            trainingData = loadCIFAR10Data("cifar-10-batches-bin/data_batch_1.bin", 10000);
            validationData = loadCIFAR10Data("cifar-10-batches-bin/test_batch.bin", 2000);
        } catch (IOException e) {
            fail("Failed to load CIFAR-10 data: " + e.getMessage());
        }

        // Create trainer
        trainer = new Trainer(network, trainingData, validationData, BATCH_SIZE, EPOCHS);
    }

    @Test
    void testCIFAR10Classification() {
        // Train the network
        trainer.train();

        // Evaluate on validation set
        double accuracy = evaluateAccuracy(validationData);
        System.out.printf("Validation Accuracy: %.2f%%%n", accuracy * 100);

        // Check if accuracy meets minimum requirement
        assertTrue(accuracy >= MIN_ACCURACY, 
                  String.format("Accuracy %.2f%% is below minimum requirement of %.2f%%", 
                               accuracy * 100, MIN_ACCURACY * 100));
    }

    @Test
    void testModelPersistence() {
        // Train the network
        trainer.train();

        // Save the model
        String modelPath = "cifar10_model.ser";
        ModelSerializer.saveModel(network, modelPath);

        // Load the model
        Network loadedNetwork = ModelSerializer.loadModel(modelPath);

        // Evaluate loaded model
        double originalAccuracy = evaluateAccuracy(validationData);
        double loadedAccuracy = evaluateAccuracy(validationData, loadedNetwork);

        // Check that loaded model performs similarly
        assertEquals(originalAccuracy, loadedAccuracy, 0.01, 
                    "Loaded model accuracy differs significantly from original model");
    }

    @Test
    void testTrainingStability() {
        // Train the network
        trainer.train();

        // Check training history
        assertFalse(trainer.getTrainingHistory().isEmpty(), "Training history is empty");
        assertFalse(trainer.getValidationHistory().isEmpty(), "Validation history is empty");

        // Check that loss decreases over time
        double[] trainingLosses = trainer.getTrainingHistory().stream()
            .mapToDouble(Double::doubleValue)
            .toArray();
        double[] validationLosses = trainer.getValidationHistory().stream()
            .mapToDouble(Double::doubleValue)
            .toArray();

        // Calculate average loss reduction
        double trainingReduction = calculateLossReduction(trainingLosses);
        double validationReduction = calculateLossReduction(validationLosses);

        // Check that losses decrease significantly
        assertTrue(trainingReduction > 0.1, "Training loss did not decrease significantly");
        assertTrue(validationReduction > 0.1, "Validation loss did not decrease significantly");
    }

    private DataSet loadCIFAR10Data(String filename, int numSamples) throws IOException {
        DataSet dataset = new DataSet(IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS, NUM_CLASSES);
        
        try (DataInputStream stream = new DataInputStream(new FileInputStream(filename))) {
            for (int i = 0; i < numSamples; i++) {
                // Read label
                int label = stream.readUnsignedByte();
                
                // Read image data
                double[] image = new double[IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS];
                for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS; j++) {
                    image[j] = stream.readUnsignedByte() / 255.0; // Normalize to [0,1]
                }

                // Create one-hot encoded target
                double[] target = new double[NUM_CLASSES];
                target[label] = 1.0;

                dataset.addSample(image, target);
            }
        }
        return dataset;
    }

    private double evaluateAccuracy(DataSet dataset) {
        return evaluateAccuracy(dataset, network);
    }

    private double evaluateAccuracy(DataSet dataset, Network network) {
        int correct = 0;
        int total = 0;

        for (int i = 0; i < dataset.size(); i++) {
            double[] input = dataset.getInputs().get(i);
            double[] target = dataset.getTargets().get(i);
            double[] output = network.forward(input);

            // Find predicted class
            int predictedClass = 0;
            double maxOutput = output[0];
            for (int j = 1; j < output.length; j++) {
                if (output[j] > maxOutput) {
                    maxOutput = output[j];
                    predictedClass = j;
                }
            }

            // Find true class
            int trueClass = 0;
            for (int j = 0; j < target.length; j++) {
                if (target[j] == 1.0) {
                    trueClass = j;
                    break;
                }
            }

            if (predictedClass == trueClass) {
                correct++;
            }
            total++;
        }

        return (double) correct / total;
    }

    private double calculateLossReduction(double[] losses) {
        if (losses.length < 2) return 0.0;
        
        double initialLoss = losses[0];
        double finalLoss = losses[losses.length - 1];
        return (initialLoss - finalLoss) / initialLoss;
    }
} 