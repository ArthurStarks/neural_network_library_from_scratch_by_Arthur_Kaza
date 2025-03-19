package com.neuralnetwork.examples.mnist;

import com.neuralnetwork.core.*;
import com.neuralnetwork.layers.*;
import com.neuralnetwork.optimizers.*;
import com.neuralnetwork.activations.*;
import com.neuralnetwork.utils.*;

import java.io.*;
import java.util.*;

public class MNISTExample {
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 10;
    private static final double LEARNING_RATE = 0.001;

    public static void main(String[] args) {
        try {
            // Charger les données MNIST
            System.out.println("Chargement des données MNIST...");
            MNISTDataLoader loader = new MNISTDataLoader();
            double[][][] trainingData = loader.loadTrainingData();
            double[][] trainingLabels = loader.loadTrainingLabels();
            double[][][] testData = loader.loadTestData();
            double[][] testLabels = loader.loadTestLabels();

            // Créer le réseau
            Network network = new Network();
            
            // Couche d'entrée (28x28 = 784 pixels)
            network.addLayer(new ConvolutionLayer(1, 32, 3, 3, Activation.RELU));
            network.addLayer(new MaxPoolingLayer(2, 2));
            
            // Première couche cachée
            network.addLayer(new ConvolutionLayer(32, 64, 3, 3, Activation.RELU));
            network.addLayer(new MaxPoolingLayer(2, 2));
            
            // Aplatir les données
            network.addLayer(new FlattenLayer());
            
            // Couches denses
            network.addLayer(new DenseLayer(1600, 128, Activation.RELU));
            network.addLayer(new DropoutLayer(0.5));
            network.addLayer(new DenseLayer(128, 10, Activation.SOFTMAX));

            // Configurer l'optimiseur
            network.setOptimizer(new AdamOptimizer(LEARNING_RATE));

            // Créer le visualiseur
            NetworkVisualizer visualizer = new NetworkVisualizer(network);

            // Entraînement
            System.out.println("Début de l'entraînement...");
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                double epochLoss = 0.0;
                int correctPredictions = 0;

                // Traitement par lots
                for (int i = 0; i < trainingData.length; i += BATCH_SIZE) {
                    // Préparer le lot
                    int batchSize = Math.min(BATCH_SIZE, trainingData.length - i);
                    double[][][] batchData = Arrays.copyOfRange(trainingData, i, i + batchSize);
                    double[][] batchLabels = Arrays.copyOfRange(trainingLabels, i, i + batchSize);

                    // Forward pass
                    double[][] predictions = network.forward(batchData);
                    
                    // Calculer la perte
                    double batchLoss = Loss.crossEntropy(predictions, batchLabels);
                    epochLoss += batchLoss;

                    // Backward pass
                    network.backward(batchLabels);

                    // Mettre à jour les paramètres
                    network.updateParameters(LEARNING_RATE);

                    // Calculer la précision
                    for (int j = 0; j < batchSize; j++) {
                        int predicted = argmax(predictions[j]);
                        int actual = argmax(batchLabels[j]);
                        if (predicted == actual) {
                            correctPredictions++;
                        }
                    }

                    // Afficher la progression
                    if ((i + batchSize) % 1000 == 0) {
                        System.out.printf("Époque %d/%d - Lot %d/%d - Perte: %.4f - Précision: %.2f%%\n",
                            epoch + 1, EPOCHS, i + batchSize, trainingData.length,
                            batchLoss, (correctPredictions * 100.0) / (i + batchSize));
                    }
                }

                // Afficher les statistiques de l'époque
                double avgLoss = epochLoss / (trainingData.length / BATCH_SIZE);
                double accuracy = (correctPredictions * 100.0) / trainingData.length;
                System.out.printf("Époque %d/%d terminée - Perte moyenne: %.4f - Précision: %.2f%%\n",
                    epoch + 1, EPOCHS, avgLoss, accuracy);

                // Mettre à jour la visualisation
                visualizer.update(epoch, avgLoss, accuracy);
            }

            // Évaluation sur l'ensemble de test
            System.out.println("\nÉvaluation sur l'ensemble de test...");
            double testLoss = 0.0;
            int testCorrect = 0;

            for (int i = 0; i < testData.length; i += BATCH_SIZE) {
                int batchSize = Math.min(BATCH_SIZE, testData.length - i);
                double[][][] batchData = Arrays.copyOfRange(testData, i, i + batchSize);
                double[][] batchLabels = Arrays.copyOfRange(testLabels, i, i + batchSize);

                double[][] predictions = network.forward(batchData);
                double batchLoss = Loss.crossEntropy(predictions, batchLabels);
                testLoss += batchLoss;

                for (int j = 0; j < batchSize; j++) {
                    int predicted = argmax(predictions[j]);
                    int actual = argmax(batchLabels[j]);
                    if (predicted == actual) {
                        testCorrect++;
                    }
                }
            }

            double testAccuracy = (testCorrect * 100.0) / testData.length;
            System.out.printf("Résultats finaux sur l'ensemble de test:\n");
            System.out.printf("Perte: %.4f\n", testLoss / (testData.length / BATCH_SIZE));
            System.out.printf("Précision: %.2f%%\n", testAccuracy);

            // Sauvegarder le modèle
            ModelSaver.save(network, "mnist_model.ser");
            System.out.println("\nModèle sauvegardé dans 'mnist_model.ser'");

        } catch (Exception e) {
            System.err.println("Erreur lors de l'exécution: " + e.getMessage());
            e.printStackTrace();
        }
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