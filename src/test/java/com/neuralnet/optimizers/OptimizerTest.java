package com.neuralnet.optimizers;

import com.neuralnet.core.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class OptimizerTest {
    private static final double EPSILON = 1e-10;
    private Connection connection;
    private Neuron neuron;
    private double learningRate;

    @BeforeEach
    void setUp() {
        neuron = new Neuron();
        connection = new Connection(new Neuron(), neuron, 1.0);
        learningRate = 0.1;
    }

    @Test
    void testSGD() {
        SGD sgd = new SGD();
        
        // Initialize
        sgd.initialize(connection);
        sgd.initialize(neuron);
        
        // Test weight update
        connection.setDeltaWeight(0.5);
        double initialWeight = connection.getWeight();
        sgd.updateWeights(connection, learningRate);
        assertEquals(initialWeight + learningRate * 0.5, connection.getWeight(), EPSILON);
        
        // Test bias update
        neuron.setDeltaBias(0.3);
        double initialBias = neuron.getBias();
        sgd.updateBias(neuron, learningRate);
        assertEquals(initialBias + learningRate * 0.3, neuron.getBias(), EPSILON);
    }

    @Test
    void testAdam() {
        Adam adam = new Adam();
        
        // Initialize
        adam.initialize(connection);
        adam.initialize(neuron);
        
        // Test weight update
        connection.setDeltaWeight(0.5);
        double initialWeight = connection.getWeight();
        adam.updateWeights(connection, learningRate);
        assertNotEquals(initialWeight, connection.getWeight());
        
        // Test bias update
        neuron.setDeltaBias(0.3);
        double initialBias = neuron.getBias();
        adam.updateBias(neuron, learningRate);
        assertNotEquals(initialBias, neuron.getBias());
    }

    @Test
    void testRMSprop() {
        RMSprop rmsprop = new RMSprop();
        
        // Initialize
        rmsprop.initialize(connection);
        rmsprop.initialize(neuron);
        
        // Test weight update
        connection.setDeltaWeight(0.5);
        double initialWeight = connection.getWeight();
        rmsprop.updateWeights(connection, learningRate);
        assertNotEquals(initialWeight, connection.getWeight());
        
        // Test bias update
        neuron.setDeltaBias(0.3);
        double initialBias = neuron.getBias();
        rmsprop.updateBias(neuron, learningRate);
        assertNotEquals(initialBias, neuron.getBias());
    }

    @Test
    void testOptimizerConvergence() {
        // Create a simple network
        Network network = new Network(new Adam());
        network.addLayer(new Layer(1, new ReLU(), new Adam()));
        network.addLayer(new Layer(1, new Sigmoid(), new Adam()));
        
        // Create training data (simple linear regression)
        DataSet trainingData = new DataSet(1, 1);
        for (int i = 0; i < 100; i++) {
            double x = Math.random() * 2 - 1; // [-1, 1]
            double y = 2 * x + 1; // Target function
            trainingData.addSample(new double[]{x}, new double[]{y});
        }
        
        // Train the network
        Trainer trainer = new Trainer(network, trainingData, null, 10, 100);
        trainer.train();
        
        // Test prediction
        double[] output = network.forward(new double[]{0.5});
        assertEquals(2.0, output[0], 0.1); // Should be close to 2 * 0.5 + 1 = 2
    }

    @Test
    void testOptimizerStability() {
        // Create a network with different optimizers
        Network network = new Network(new Adam());
        network.addLayer(new Layer(2, new ReLU(), new Adam()));
        network.addLayer(new Layer(2, new ReLU(), new SGD()));
        network.addLayer(new Layer(1, new Sigmoid(), new RMSprop()));
        
        // Create training data (XOR problem)
        DataSet trainingData = new DataSet(2, 1);
        trainingData.addSample(new double[]{0, 0}, new double[]{0});
        trainingData.addSample(new double[]{0, 1}, new double[]{1});
        trainingData.addSample(new double[]{1, 0}, new double[]{1});
        trainingData.addSample(new double[]{1, 1}, new double[]{0});
        
        // Train the network
        Trainer trainer = new Trainer(network, trainingData, null, 4, 1000);
        trainer.train();
        
        // Test predictions
        double[] output1 = network.forward(new double[]{0, 0});
        double[] output2 = network.forward(new double[]{0, 1});
        double[] output3 = network.forward(new double[]{1, 0});
        double[] output4 = network.forward(new double[]{1, 1});
        
        // Check XOR logic
        assertTrue(output1[0] < 0.5);
        assertTrue(output2[0] > 0.5);
        assertTrue(output3[0] > 0.5);
        assertTrue(output4[0] < 0.5);
    }
} 