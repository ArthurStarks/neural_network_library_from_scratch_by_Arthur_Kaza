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

public class FashionMNISTTest {
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_CHANNELS = 1;
    private static final int NUM_CLASSES = 10;
    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 5;
    private static final double MIN_ACCURACY = 0.85; // Précision minimale attendue sur Fashion MNIST

    private Network network;
    private DataSet trainingData;
    private DataSet validationData;
    private Optimizer optimizer;
    private Trainer trainer;

    @BeforeEach
    void setUp() {
        // Créer un réseau avec une architecture de type CNN
        optimizer = new Adam();
        network = new Network(optimizer);
        
        // Ajouter des couches
        network.addLayer(new Layer(IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS, new ReLU(), optimizer));  // Couche d'entrée
        network.addLayer(new Layer(512, new ReLU(), optimizer));                                     // Couche cachée
        network.addLayer(new Layer(256, new ReLU(), optimizer));                                     // Couche cachée
        network.addLayer(new Layer(128, new ReLU(), optimizer));                                     // Couche cachée
        network.addLayer(new Layer(NUM_CLASSES, new Softmax(), optimizer));                         // Couche de sortie

        // Charger les données Fashion MNIST
        try {
            trainingData = loadFashionMNISTData("fashion-mnist/train-images-idx3-ubyte", 
                                               "fashion-mnist/train-labels-idx1-ubyte", 
                                               60000);
            validationData = loadFashionMNISTData("fashion-mnist/t10k-images-idx3-ubyte", 
                                                "fashion-mnist/t10k-labels-idx1-ubyte", 
                                                10000);
        } catch (IOException e) {
            fail("Échec du chargement des données Fashion MNIST : " + e.getMessage());
        }

        // Créer l'entraîneur
        trainer = new Trainer(network, trainingData, validationData, BATCH_SIZE, EPOCHS);
    }

    @Test
    void testFashionMNISTClassification() {
        // Entraîner le réseau
        trainer.train();

        // Évaluer sur l'ensemble de validation
        double accuracy = evaluateAccuracy(validationData);
        System.out.printf("Précision de Validation : %.2f%%%n", accuracy * 100);

        // Vérifier si la précision atteint le minimum requis
        assertTrue(accuracy >= MIN_ACCURACY, 
                  String.format("La précision %.2f%% est inférieure au minimum requis de %.2f%%", 
                               accuracy * 100, MIN_ACCURACY * 100));
    }

    @Test
    void testModelPersistence() {
        // Entraîner le réseau
        trainer.train();

        // Sauvegarder le modèle
        String modelPath = "fashion_mnist_model.ser";
        ModelSerializer.saveModel(network, modelPath);

        // Charger le modèle
        Network loadedNetwork = ModelSerializer.loadModel(modelPath);

        // Évaluer le modèle chargé
        double originalAccuracy = evaluateAccuracy(validationData);
        double loadedAccuracy = evaluateAccuracy(validationData, loadedNetwork);

        // Vérifier que le modèle chargé performe de manière similaire
        assertEquals(originalAccuracy, loadedAccuracy, 0.01, 
                    "La précision du modèle chargé diffère significativement du modèle original");
    }

    @Test
    void testTrainingStability() {
        // Entraîner le réseau
        trainer.train();

        // Vérifier l'historique d'entraînement
        assertFalse(trainer.getTrainingHistory().isEmpty(), "L'historique d'entraînement est vide");
        assertFalse(trainer.getValidationHistory().isEmpty(), "L'historique de validation est vide");

        // Vérifier que la perte diminue au fil du temps
        double[] trainingLosses = trainer.getTrainingHistory().stream()
            .mapToDouble(Double::doubleValue)
            .toArray();
        double[] validationLosses = trainer.getValidationHistory().stream()
            .mapToDouble(Double::doubleValue)
            .toArray();

        // Calculer la réduction moyenne de la perte
        double trainingReduction = calculateLossReduction(trainingLosses);
        double validationReduction = calculateLossReduction(validationLosses);

        // Vérifier que les pertes diminuent significativement
        assertTrue(trainingReduction > 0.1, "La perte d'entraînement n'a pas diminué significativement");
        assertTrue(validationReduction > 0.1, "La perte de validation n'a pas diminué significativement");
    }

    @Test
    void testBatchProcessing() {
        // Tester avec différentes tailles de lots
        int[] batchSizes = {16, 32, 64, 128};
        double[] accuracies = new double[batchSizes.length];

        for (int i = 0; i < batchSizes.length; i++) {
            Network batchNetwork = new Network(optimizer);
            batchNetwork.addLayer(new Layer(IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS, new ReLU(), optimizer));
            batchNetwork.addLayer(new Layer(512, new ReLU(), optimizer));
            batchNetwork.addLayer(new Layer(256, new ReLU(), optimizer));
            batchNetwork.addLayer(new Layer(128, new ReLU(), optimizer));
            batchNetwork.addLayer(new Layer(NUM_CLASSES, new Softmax(), optimizer));

            Trainer batchTrainer = new Trainer(batchNetwork, trainingData, validationData, 
                                             batchSizes[i], EPOCHS);
            batchTrainer.train();
            accuracies[i] = evaluateAccuracy(validationData, batchNetwork);
        }

        // Vérifier que toutes les tailles de lots atteignent une précision raisonnable
        for (double accuracy : accuracies) {
            assertTrue(accuracy >= MIN_ACCURACY * 0.9, 
                      "La précision du traitement par lots est inférieure au minimum requis");
        }
    }

    private DataSet loadFashionMNISTData(String imagesFile, String labelsFile, int numSamples) 
            throws IOException {
        DataSet dataset = new DataSet(IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS, NUM_CLASSES);
        
        try (DataInputStream imageStream = new DataInputStream(new FileInputStream(imagesFile));
             DataInputStream labelStream = new DataInputStream(new FileInputStream(labelsFile))) {
            
            // Ignorer l'en-tête
            imageStream.readInt(); // Nombre magique
            imageStream.readInt(); // Nombre d'images
            imageStream.readInt(); // Nombre de lignes
            imageStream.readInt(); // Nombre de colonnes
            
            labelStream.readInt(); // Nombre magique
            labelStream.readInt(); // Nombre d'éléments
            
            for (int i = 0; i < numSamples; i++) {
                // Lire l'étiquette
                int label = labelStream.readUnsignedByte();
                
                // Lire les données de l'image
                double[] image = new double[IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS];
                for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS; j++) {
                    image[j] = imageStream.readUnsignedByte() / 255.0; // Normaliser à [0,1]
                }

                // Créer l'encodage one-hot de la cible
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

            // Trouver la classe prédite
            int predictedClass = 0;
            double maxOutput = output[0];
            for (int j = 1; j < output.length; j++) {
                if (output[j] > maxOutput) {
                    maxOutput = output[j];
                    predictedClass = j;
                }
            }

            // Trouver la classe réelle
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