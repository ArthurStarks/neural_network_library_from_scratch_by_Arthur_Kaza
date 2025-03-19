package com.neuralnetwork.examples.mnist;

import com.neuralnetwork.core.*;
import com.neuralnetwork.layers.*;
import com.neuralnetwork.optimizers.*;
import com.neuralnetwork.activations.*;
import com.neuralnetwork.training.*;

import javax.swing.*;

public class MNISTExample {
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;
    private static final double LEARNING_RATE = 0.001;

    public static void main(String[] args) {
        try {
            // Charger les données MNIST
            System.out.println("Chargement des données MNIST...");
            MNISTDataLoader loader = new MNISTDataLoader();
            loader.downloadMNISTData();
            
            double[][][] trainData = loader.loadTrainingData();
            double[][] trainLabels = loader.loadTrainingLabels();
            double[][][] testData = loader.loadTestData();
            double[][] testLabels = loader.loadTestLabels();

            // Créer le réseau de neurones
            System.out.println("Création du réseau de neurones...");
            Network network = new Network();
            
            // Ajouter les couches
            network.addLayer(new ConvolutionalLayer(1, 32, 3, 3, 1, 1)); // Couche de convolution
            network.addLayer(new ReLU());
            network.addLayer(new MaxPoolingLayer(2, 2, 2, 2)); // Couche de pooling
            network.addLayer(new ConvolutionalLayer(32, 64, 3, 3, 1, 1));
            network.addLayer(new ReLU());
            network.addLayer(new MaxPoolingLayer(2, 2, 2, 2));
            network.addLayer(new FlattenLayer());
            network.addLayer(new DenseLayer(64 * 7 * 7, 128));
            network.addLayer(new ReLU());
            network.addLayer(new DropoutLayer(0.5));
            network.addLayer(new DenseLayer(128, 10));
            network.addLayer(new Softmax());

            // Configurer l'optimiseur
            Optimizer optimizer = new Adam(LEARNING_RATE);
            network.setOptimizer(optimizer);

            // Créer le trainer
            Trainer trainer = new Trainer(network, BATCH_SIZE);

            // Créer et afficher le visualiseur
            TrainingVisualizer visualizer = new TrainingVisualizer();
            SwingUtilities.invokeLater(() -> visualizer.setVisible(true));

            // Entraîner le réseau
            System.out.println("Début de l'entraînement...");
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                System.out.println("Epoch " + (epoch + 1) + "/" + EPOCHS);
                
                // Entraînement
                double trainLoss = trainer.train(trainData, trainLabels);
                
                // Évaluation
                double testLoss = trainer.evaluate(testData, testLabels);
                double accuracy = calculateAccuracy(network, testData, testLabels);
                
                // Mettre à jour le visualiseur
                final double finalTrainLoss = trainLoss;
                final double finalTestLoss = testLoss;
                final double finalAccuracy = accuracy;
                SwingUtilities.invokeLater(() -> 
                    visualizer.update(finalTrainLoss, finalTestLoss, finalAccuracy)
                );
                
                System.out.printf("Train Loss: %.4f, Test Loss: %.4f, Accuracy: %.2f%%\n",
                                trainLoss, testLoss, accuracy * 100);
            }

            // Sauvegarder le modèle
            System.out.println("Sauvegarde du modèle...");
            ModelSerializer.save(network, "mnist_model.ser");

            // Tester le modèle sur quelques exemples
            System.out.println("\nTest sur quelques exemples :");
            for (int i = 0; i < 5; i++) {
                double[] prediction = network.forward(testData[i]);
                int predictedDigit = argmax(prediction);
                int actualDigit = argmax(testLabels[i]);
                System.out.printf("Exemple %d: Prédit = %d, Réel = %d\n",
                                i + 1, predictedDigit, actualDigit);
            }

        } catch (Exception e) {
            System.err.println("Erreur lors de l'exécution : " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static double calculateAccuracy(Network network, double[][][] testData, double[][] testLabels) {
        int correct = 0;
        for (int i = 0; i < testData.length; i++) {
            double[] prediction = network.forward(testData[i]);
            int predictedDigit = argmax(prediction);
            int actualDigit = argmax(testLabels[i]);
            if (predictedDigit == actualDigit) {
                correct++;
            }
        }
        return (double) correct / testData.length;
    }

    private static int argmax(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
} 