package com.neuralnetwork.layers;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class DenseLayerTest {
    private DenseLayer layer;
    private double[] input;
    private double[] expectedOutput;
    private double[] expectedGradient;

    @Before
    public void setUp() {
        // Créer une couche dense avec 3 entrées et 2 sorties
        layer = new DenseLayer(3, 2, Activation.RELU);
        
        // Données de test
        input = new double[]{1.0, 2.0, 3.0};
        expectedOutput = new double[]{0.5, 1.0}; // Valeurs attendues après ReLU
        expectedGradient = new double[]{0.1, 0.2}; // Gradients attendus
    }

    @Test
    public void testForwardPropagation() {
        double[] output = layer.forward(input);
        
        // Vérifier la dimension de la sortie
        assertEquals(2, output.length);
        
        // Vérifier que les valeurs sont positives (ReLU)
        for (double value : output) {
            assertTrue(value >= 0);
        }
    }

    @Test
    public void testBackwardPropagation() {
        // Forward pass
        double[] output = layer.forward(input);
        
        // Backward pass
        double[] gradient = layer.backward(expectedGradient);
        
        // Vérifier la dimension du gradient
        assertEquals(3, gradient.length);
        
        // Vérifier que le gradient n'est pas nul
        for (double value : gradient) {
            assertNotEquals(0.0, value);
        }
    }

    @Test
    public void testWeightInitialization() {
        double[][] weights = layer.getWeights();
        
        // Vérifier les dimensions des poids
        assertEquals(2, weights.length); // 2 neurones de sortie
        assertEquals(3, weights[0].length); // 3 entrées
        
        // Vérifier que les poids sont initialisés avec des valeurs non nulles
        for (double[] row : weights) {
            for (double weight : row) {
                assertNotEquals(0.0, weight);
            }
        }
    }

    @Test
    public void testBiasInitialization() {
        double[] biases = layer.getBiases();
        
        // Vérifier la dimension des biais
        assertEquals(2, biases.length); // 2 neurones de sortie
        
        // Vérifier que les biais sont initialisés avec des valeurs non nulles
        for (double bias : biases) {
            assertNotEquals(0.0, bias);
        }
    }

    @Test
    public void testUpdateParameters() {
        // Forward pass
        double[] output = layer.forward(input);
        
        // Backward pass
        double[] gradient = layer.backward(expectedGradient);
        
        // Sauvegarder les anciens paramètres
        double[][] oldWeights = layer.getWeights().clone();
        double[] oldBiases = layer.getBiases().clone();
        
        // Mettre à jour les paramètres
        layer.updateParameters(0.01); // Learning rate = 0.01
        
        // Vérifier que les paramètres ont été mis à jour
        double[][] newWeights = layer.getWeights();
        double[] newBiases = layer.getBiases();
        
        for (int i = 0; i < newWeights.length; i++) {
            for (int j = 0; j < newWeights[i].length; j++) {
                assertNotEquals(oldWeights[i][j], newWeights[i][j]);
            }
            assertNotEquals(oldBiases[i], newBiases[i]);
        }
    }

    @Test
    public void testActivationFunction() {
        // Tester avec des entrées négatives (devraient être mises à zéro par ReLU)
        double[] negativeInput = new double[]{-1.0, -2.0, -3.0};
        double[] output = layer.forward(negativeInput);
        
        for (double value : output) {
            assertEquals(0.0, value, 1e-10);
        }
    }
} 