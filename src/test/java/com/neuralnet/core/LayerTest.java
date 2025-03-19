package com.neuralnet.core;

import com.neuralnet.activations.ReLU;
import com.neuralnet.optimizers.Adam;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LayerTest {
    private Layer layer;
    private Layer inputLayer;
    private Layer outputLayer;
    private Optimizer optimizer;

    @BeforeEach
    void setUp() {
        optimizer = new Adam();
        layer = new Layer(3, new ReLU(), optimizer);
        inputLayer = new Layer(2, new ReLU(), optimizer);
        outputLayer = new Layer(1, new ReLU(), optimizer);
    }

    @Test
    void testInitialization() {
        assertEquals(3, layer.getNeurons().size());
        assertTrue(layer.getNeurons().get(0).getInputConnections().isEmpty());
        assertTrue(layer.getNeurons().get(0).getOutputConnections().isEmpty());
    }

    @Test
    void testConnectTo() {
        layer.connectTo(outputLayer);
        
        // Check connections
        assertEquals(3, layer.getNeurons().get(0).getOutputConnections().size());
        assertEquals(1, outputLayer.getNeurons().get(0).getInputConnections().size());
    }

    @Test
    void testForward() {
        // Set up network
        inputLayer.connectTo(layer);
        layer.connectTo(outputLayer);

        // Set input values
        double[] input = {1.0, 0.5};
        inputLayer.setInputs(input);

        // Forward pass
        layer.forward();

        // Check outputs
        assertTrue(layer.getNeurons().get(0).getOutput() > 0);
        assertTrue(layer.getNeurons().get(1).getOutput() > 0);
        assertTrue(layer.getNeurons().get(2).getOutput() > 0);
    }

    @Test
    void testBackward() {
        // Set up network
        inputLayer.connectTo(layer);
        layer.connectTo(outputLayer);

        // Set input values
        double[] input = {1.0, 0.5};
        inputLayer.setInputs(input);

        // Forward pass
        layer.forward();
        outputLayer.forward();

        // Set output layer deltas
        outputLayer.getNeurons().get(0).setDelta(1.0);

        // Backward pass
        layer.backward();

        // Check deltas
        assertTrue(layer.getNeurons().get(0).getDelta() != 0);
        assertTrue(layer.getNeurons().get(1).getDelta() != 0);
        assertTrue(layer.getNeurons().get(2).getDelta() != 0);
    }

    @Test
    void testUpdateWeights() {
        // Set up network
        inputLayer.connectTo(layer);
        layer.connectTo(outputLayer);

        // Set input values and perform forward pass
        double[] input = {1.0, 0.5};
        inputLayer.setInputs(input);
        layer.forward();
        outputLayer.forward();

        // Set output layer deltas and perform backward pass
        outputLayer.getNeurons().get(0).setDelta(1.0);
        layer.backward();

        // Store initial weights
        double[] initialWeights = new double[3];
        for (int i = 0; i < 3; i++) {
            initialWeights[i] = layer.getNeurons().get(i).getInputConnections().get(0).getWeight();
        }

        // Update weights
        layer.updateWeights(0.1);

        // Check weights have changed
        for (int i = 0; i < 3; i++) {
            assertNotEquals(initialWeights[i], 
                          layer.getNeurons().get(i).getInputConnections().get(0).getWeight());
        }
    }

    @Test
    void testReset() {
        // Set up network and perform some operations
        inputLayer.connectTo(layer);
        layer.connectTo(outputLayer);
        double[] input = {1.0, 0.5};
        inputLayer.setInputs(input);
        layer.forward();
        outputLayer.forward();
        outputLayer.getNeurons().get(0).setDelta(1.0);
        layer.backward();

        // Reset layer
        layer.reset();

        // Check reset values
        for (Neuron neuron : layer.getNeurons()) {
            assertEquals(0.0, neuron.getOutput());
            assertEquals(0.0, neuron.getDelta());
            assertEquals(0.0, neuron.getDeltaBias());
        }
    }
} 