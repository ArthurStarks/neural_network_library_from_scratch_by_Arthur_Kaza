package com.neuralnet.examples;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import com.neuralnet.training.*;
import com.neuralnet.layers.*;
import java.io.*;
import java.util.Random;

/**
 * Example demonstrating MNIST digit classification using our neural network library.
 * This example shows how to:
 * 1. Load and preprocess MNIST data
 * 2. Create a CNN architecture
 * 3. Train the network
 * 4. Evaluate performance
 */
public class MNISTExample {
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_CLASSES = 10;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;

    public static void main(String[] args) {
        try {
            // Load MNIST data
            System.out.println("Loading MNIST data...");
            DataSet trainingData = loadMNISTData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 60000);
            DataSet validationData = loadMNISTData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000);

            // Create network
            Network network = new Network();
            
            // Add layers
            network.addLayer(new ConvolutionalLayer(IMAGE_SIZE, IMAGE_SIZE, 1, 3, 1, 32, new ReLU()));
            network.addLayer(new BatchNormalizationLayer(32, 1e-5, 0.9, new ReLU()));
            network.addLayer(new ConvolutionalLayer(26, 26, 32, 3, 1, 64, new ReLU()));
            network.addLayer(new BatchNormalizationLayer(64, 1e-5, 0.9, new ReLU()));
            network.addLayer(new Layer(64 * 24 * 24, new ReLU()));
            network.addLayer(new DropoutLayer(64 * 24 * 24, 0.5, new ReLU()));
            network.addLayer(new Layer(NUM_CLASSES, new Softmax()));

            // Create optimizer
            Optimizer optimizer = new Adam(0.9, 0.999, 1e-8);

            // Create trainer
            Trainer trainer = new Trainer(network, trainingData, validationData, BATCH_SIZE, EPOCHS);

            // Train the network
            System.out.println("Training network...");
            trainer.train();

            // Evaluate the network
            System.out.println("\nEvaluating network...");
            evaluateNetwork(network, validationData);

            // Save the model
            System.out.println("\nSaving model...");
            ModelSerializer.saveModel(network, "mnist_model.ser");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static DataSet loadMNISTData(String imagesFile, String labelsFile, int numSamples) 
            throws IOException {
        DataSet dataset = new DataSet(IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES);
        
        try (DataInputStream imageStream = new DataInputStream(new FileInputStream(imagesFile));
             DataInputStream labelStream = new DataInputStream(new FileInputStream(labelsFile))) {
            
            // Skip headers
            imageStream.readInt(); // Magic number
            imageStream.readInt(); // Number of images
            imageStream.readInt(); // Number of rows
            imageStream.readInt(); // Number of columns
            
            labelStream.readInt(); // Magic number
            labelStream.readInt(); // Number of items

            // Read data
            for (int i = 0; i < numSamples; i++) {
                // Read image
                double[] image = new double[IMAGE_SIZE * IMAGE_SIZE];
                for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE; j++) {
                    image[j] = imageStream.readUnsignedByte() / 255.0; // Normalize to [0,1]
                }

                // Read label and create one-hot encoding
                int label = labelStream.readUnsignedByte();
                double[] target = new double[NUM_CLASSES];
                target[label] = 1.0;

                dataset.addSample(image, target);
            }
        }
        return dataset;
    }

    private static void evaluateNetwork(Network network, DataSet validationData) {
        int correct = 0;
        int total = 0;

        for (int i = 0; i < validationData.size(); i++) {
            double[] input = validationData.getInputs().get(i);
            double[] target = validationData.getTargets().get(i);
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

        double accuracy = (double) correct / total * 100;
        System.out.printf("Validation Accuracy: %.2f%%%n", accuracy);
    }
} 