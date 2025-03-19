package com.neuralnet.tests;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

public class LayerTest {
    private Layer layer;
    private Optimizer optimizer;
    private static final double EPSILON = 1e-10;

    @BeforeEach
    void setUp() {
        optimizer = new Adam();
        layer = new Layer(3, new ReLU(), optimizer);
    }

    @Test
    void testLayerInitialization() {
        assertEquals(3, layer.getNeurons().size(), "La couche devrait avoir le bon nombre de neurones");
        assertNotNull(layer.getOptimizer(), "La couche devrait avoir un optimiseur");
        assertNotNull(layer.getActivation(), "La couche devrait avoir une fonction d'activation");
    }

    @Test
    void testForwardPass() {
        double[] input = {1.0, 2.0, 3.0};
        double[] output = layer.forward(input);

        assertEquals(3, output.length, "La sortie devrait avoir la même taille que l'entrée");
        assertTrue(output[0] >= 0, "La sortie ReLU devrait être non négative");
    }

    @Test
    void testBackwardPass() {
        double[] input = {1.0, 2.0, 3.0};
        double[] target = {0.5, 1.0, 1.5};
        
        // Passe avant
        double[] output = layer.forward(input);
        
        // Passe arrière
        layer.backward(target);

        // Vérifier que les gradients sont calculés
        for (Neuron neuron : layer.getNeurons()) {
            assertNotEquals(0.0, neuron.getDeltaBias(), "Le neurone devrait avoir un gradient de biais non nul");
        }
    }

    @Test
    void testWeightUpdates() {
        double[] input = {1.0, 2.0, 3.0};
        double[] target = {0.5, 1.0, 1.5};
        double learningRate = 0.1;

        // Stocker les poids initiaux
        List<Double> initialWeights = layer.getNeurons().get(0).getWeights();
        
        // Passe avant et arrière
        layer.forward(input);
        layer.backward(target);
        layer.updateWeights(learningRate);

        // Vérifier que les poids ont été mis à jour
        List<Double> updatedWeights = layer.getNeurons().get(0).getWeights();
        for (int i = 0; i < initialWeights.size(); i++) {
            assertNotEquals(initialWeights.get(i), updatedWeights.get(i), 
                          "Les poids devraient être mis à jour après l'entraînement");
        }
    }

    @Test
    void testDifferentActivationFunctions() {
        // Test avec Sigmoid
        Layer sigmoidLayer = new Layer(3, new Sigmoid(), optimizer);
        double[] input = {1.0, 2.0, 3.0};
        double[] output = sigmoidLayer.forward(input);
        for (double value : output) {
            assertTrue(value > 0 && value < 1, "La sortie Sigmoid devrait être entre 0 et 1");
        }

        // Test avec Tanh
        Layer tanhLayer = new Layer(3, new Tanh(), optimizer);
        output = tanhLayer.forward(input);
        for (double value : output) {
            assertTrue(value > -1 && value < 1, "La sortie Tanh devrait être entre -1 et 1");
        }
    }

    @Test
    void testLayerConnections() {
        Layer nextLayer = new Layer(2, new ReLU(), optimizer);
        layer.connectTo(nextLayer);

        assertEquals(2, layer.getNeurons().get(0).getOutputConnections().size(),
                    "Chaque neurone devrait avoir des connexions vers la couche suivante");
        assertEquals(3, nextLayer.getNeurons().get(0).getInputConnections().size(),
                    "Chaque neurone de la couche suivante devrait avoir des connexions depuis la couche précédente");
    }

    @Test
    void testGradientFlow() {
        Layer nextLayer = new Layer(2, new ReLU(), optimizer);
        layer.connectTo(nextLayer);

        double[] input = {1.0, 2.0, 3.0};
        double[] target = {0.5, 1.0};

        // Passe avant à travers les deux couches
        double[] layerOutput = layer.forward(input);
        double[] nextLayerOutput = nextLayer.forward(layerOutput);

        // Passe arrière
        nextLayer.backward(target);
        layer.backward(nextLayer.getGradients());

        // Vérifier que les gradients circulent correctement
        for (Neuron neuron : layer.getNeurons()) {
            assertNotEquals(0.0, neuron.getDeltaBias(), "Les neurones devraient recevoir des gradients");
        }
    }

    @Test
    void testLayerReset() {
        double[] input = {1.0, 2.0, 3.0};
        double[] target = {0.5, 1.0, 1.5};

        // Passe avant et arrière
        layer.forward(input);
        layer.backward(target);

        // Réinitialiser la couche
        layer.reset();

        // Vérifier que tous les gradients sont nuls
        for (Neuron neuron : layer.getNeurons()) {
            assertEquals(0.0, neuron.getDeltaBias(), "Le gradient de biais du neurone devrait être réinitialisé");
            for (Connection conn : neuron.getInputConnections()) {
                assertEquals(0.0, conn.getDeltaWeight(), "Le gradient de poids de la connexion devrait être réinitialisé");
            }
        }
    }

    @Test
    void testLayerSerialization() {
        // Créer une couche avec un état
        double[] input = {1.0, 2.0, 3.0};
        layer.forward(input);
        layer.backward(new double[]{0.5, 1.0, 1.5});

        // Stocker l'état initial
        List<Double> initialWeights = layer.getNeurons().get(0).getWeights();
        List<Double> initialBiases = layer.getNeurons().stream()
            .map(Neuron::getBias)
            .toList();

        // Sérialiser et désérialiser
        Layer deserializedLayer = ModelSerializer.clone(layer);

        // Vérifier que l'état est préservé
        List<Double> deserializedWeights = deserializedLayer.getNeurons().get(0).getWeights();
        List<Double> deserializedBiases = deserializedLayer.getNeurons().stream()
            .map(Neuron::getBias)
            .toList();

        for (int i = 0; i < initialWeights.size(); i++) {
            assertEquals(initialWeights.get(i), deserializedWeights.get(i), EPSILON,
                        "Les poids devraient être préservés après la sérialisation");
        }

        for (int i = 0; i < initialBiases.size(); i++) {
            assertEquals(initialBiases.get(i), deserializedBiases.get(i), EPSILON,
                        "Les biais devraient être préservés après la sérialisation");
        }
    }
} 