package com.neuralnet.core;

import com.neuralnet.activations.ReLU;
import com.neuralnet.activations.Sigmoid;
import com.neuralnet.optimizers.Adam;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class NetworkTest {
    private Network network;
    private Optimizer optimizer;

    @BeforeEach
    void setUp() {
        optimizer = new Adam();
        network = new Network(optimizer);
    }

    @Test
    void testInitialization() {
        assertTrue(network.getLayers().isEmpty());
    }

    @Test
    void testAddLayer() {
        Layer inputLayer = new Layer(2, new ReLU(), optimizer);
        Layer hiddenLayer = new Layer(3, new ReLU(), optimizer);
        Layer outputLayer = new Layer(1, new Sigmoid(), optimizer);

        network.addLayer(inputLayer);
        network.addLayer(hiddenLayer);
        network.addLayer(outputLayer);

        assertEquals(3, network.getLayers().size());
        assertEquals(inputLayer, network.getLayers().get(0));
        assertEquals(hiddenLayer, network.getLayers().get(1));
        assertEquals(outputLayer, network.getLayers().get(2));
    }

    @Test
    void testForward() {
        // Create layers
        Layer inputLayer = new Layer(2, new ReLU(), optimizer);
        Layer hiddenLayer = new Layer(3, new ReLU(), optimizer);
        Layer outputLayer = new Layer(1, new Sigmoid(), optimizer);

        // Add layers to network
        network.addLayer(inputLayer);
        network.addLayer(hiddenLayer);
        network.addLayer(outputLayer);

        // Test forward pass
        double[] input = {1.0, 0.5};
        double[] output = network.forward(input);

        // Check output
        assertEquals(1, output.length);
        assertTrue(output[0] >= 0 && output[0] <= 1); // Sigmoid output range
    }

    @Test
    void testBackward() {
        // Create layers
        Layer inputLayer = new Layer(2, new ReLU(), optimizer);
        Layer hiddenLayer = new Layer(3, new ReLU(), optimizer);
        Layer outputLayer = new Layer(1, new Sigmoid(), optimizer);

        // Add layers to network
        network.addLayer(inputLayer);
        network.addLayer(hiddenLayer);
        network.addLayer(outputLayer);

        // Perform forward pass
        double[] input = {1.0, 0.5};
        double[] target = {0.8};
        network.forward(input);

        // Perform backward pass
        network.backward(target);

        // Check that deltas are computed
        for (Layer layer : network.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                assertTrue(neuron.getDelta() != 0);
            }
        }
    }

    @Test
    void testUpdateWeights() {
        // Create layers
        Layer inputLayer = new Layer(2, new ReLU(), optimizer);
        Layer hiddenLayer = new Layer(3, new ReLU(), optimizer);
        Layer outputLayer = new Layer(1, new Sigmoid(), optimizer);

        // Add layers to network
        network.addLayer(inputLayer);
        network.addLayer(hiddenLayer);
        network.addLayer(outputLayer);

        // Store initial weights
        double[][] initialWeights = new double[3][];
        for (int i = 0; i < 3; i++) {
            Layer layer = network.getLayers().get(i);
            initialWeights[i] = new double[layer.getNeurons().size()];
            for (int j = 0; j < layer.getNeurons().size(); j++) {
                if (!layer.getNeurons().get(j).getInputConnections().isEmpty()) {
                    initialWeights[i][j] = layer.getNeurons().get(j).getInputConnections().get(0).getWeight();
                }
            }
        }

        // Perform forward and backward pass
        double[] input = {1.0, 0.5};
        double[] target = {0.8};
        network.forward(input);
        network.backward(target);
        network.updateWeights(0.1);

        // Check weights have changed
        for (int i = 0; i < 3; i++) {
            Layer layer = network.getLayers().get(i);
            for (int j = 0; j < layer.getNeurons().size(); j++) {
                if (!layer.getNeurons().get(j).getInputConnections().isEmpty()) {
                    assertNotEquals(initialWeights[i][j],
                                  layer.getNeurons().get(j).getInputConnections().get(0).getWeight());
                }
            }
        }
    }

    @Test
    void testReset() {
        // Create layers
        Layer inputLayer = new Layer(2, new ReLU(), optimizer);
        Layer hiddenLayer = new Layer(3, new ReLU(), optimizer);
        Layer outputLayer = new Layer(1, new Sigmoid(), optimizer);

        // Add layers to network
        network.addLayer(inputLayer);
        network.addLayer(hiddenLayer);
        network.addLayer(outputLayer);

        // Perform some operations
        double[] input = {1.0, 0.5};
        double[] target = {0.8};
        network.forward(input);
        network.backward(target);

        // Reset network
        network.reset();

        // Check reset values
        for (Layer layer : network.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                assertEquals(0.0, neuron.getOutput());
                assertEquals(0.0, neuron.getDelta());
                assertEquals(0.0, neuron.getDeltaBias());
            }
        }
    }
} 