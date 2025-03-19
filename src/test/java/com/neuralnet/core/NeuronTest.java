package com.neuralnet.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class NeuronTest {
    private Neuron neuron;
    private Neuron input1;
    private Neuron input2;

    @BeforeEach
    void setUp() {
        neuron = new Neuron();
        input1 = new Neuron();
        input2 = new Neuron();
    }

    @Test
    void testInitialization() {
        assertEquals(0.0, neuron.getBias());
        assertEquals(0.0, neuron.getOutput());
        assertEquals(0.0, neuron.getDelta());
        assertEquals(0.0, neuron.getDeltaBias());
        assertTrue(neuron.getInputConnections().isEmpty());
        assertTrue(neuron.getOutputConnections().isEmpty());
    }

    @Test
    void testAddConnection() {
        Connection conn1 = new Connection(input1, neuron, 0.5);
        Connection conn2 = new Connection(input2, neuron, 0.3);

        neuron.addInputConnection(conn1);
        neuron.addInputConnection(conn2);

        assertEquals(2, neuron.getInputConnections().size());
        assertTrue(neuron.getInputConnections().contains(conn1));
        assertTrue(neuron.getInputConnections().contains(conn2));
    }

    @Test
    void testComputeOutput() {
        // Set up connections
        Connection conn1 = new Connection(input1, neuron, 0.5);
        Connection conn2 = new Connection(input2, neuron, 0.3);
        neuron.addInputConnection(conn1);
        neuron.addInputConnection(conn2);

        // Set input values
        input1.setOutput(1.0);
        input2.setOutput(0.5);

        // Compute output
        neuron.computeOutput();

        // Expected: (1.0 * 0.5 + 0.5 * 0.3 + 0.0) = 0.65
        assertEquals(0.65, neuron.getOutput(), 1e-10);
    }

    @Test
    void testComputeDelta() {
        // Set up connections
        Neuron output = new Neuron();
        Connection conn = new Connection(neuron, output, 0.5);
        neuron.addOutputConnection(conn);

        // Set values
        output.setDelta(1.0);
        neuron.setOutput(0.5);

        // Compute delta
        neuron.computeDelta();

        // Expected: 1.0 * 0.5 * 0.5 * (1 - 0.5) = 0.125
        assertEquals(0.125, neuron.getDelta(), 1e-10);
    }

    @Test
    void testUpdateWeights() {
        // Set up connections
        Connection conn1 = new Connection(input1, neuron, 0.5);
        Connection conn2 = new Connection(input2, neuron, 0.3);
        neuron.addInputConnection(conn1);
        neuron.addInputConnection(conn2);

        // Set delta weights
        conn1.setDeltaWeight(0.1);
        conn2.setDeltaWeight(0.2);
        neuron.setDeltaBias(0.3);

        // Update weights
        neuron.updateWeights(0.1); // learning rate = 0.1

        // Check updated weights
        assertEquals(0.51, conn1.getWeight(), 1e-10); // 0.5 + 0.1 * 0.1
        assertEquals(0.32, conn2.getWeight(), 1e-10); // 0.3 + 0.1 * 0.2
        assertEquals(0.03, neuron.getBias(), 1e-10);  // 0.0 + 0.1 * 0.3
    }

    @Test
    void testReset() {
        // Set some values
        neuron.setOutput(0.5);
        neuron.setDelta(0.3);
        neuron.setDeltaBias(0.2);

        // Reset neuron
        neuron.reset();

        // Check reset values
        assertEquals(0.0, neuron.getOutput());
        assertEquals(0.0, neuron.getDelta());
        assertEquals(0.0, neuron.getDeltaBias());
    }
} 