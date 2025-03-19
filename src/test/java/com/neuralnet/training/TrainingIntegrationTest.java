package com.neuralnet.training;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TrainingIntegrationTest {
    private Network network;
    private DataSet trainingData;
    private DataSet validationData;
    private Optimizer optimizer;
    private Trainer trainer;

    @BeforeEach
    void setUp() {
        // Create network
        optimizer = new Adam();
        network = new Network(optimizer);
        
        // Add layers
        network.addLayer(new Layer(2, new ReLU(), optimizer));     // Input layer
        network.addLayer(new Layer(4, new ReLU(), optimizer));     // Hidden layer
        network.addLayer(new Layer(1, new Sigmoid(), optimizer));  // Output layer

        // Create XOR training data
        trainingData = new DataSet(2, 1);
        trainingData.addSample(new double[]{0, 0}, new double[]{0});
        trainingData.addSample(new double[]{0, 1}, new double[]{1});
        trainingData.addSample(new double[]{1, 0}, new double[]{1});
        trainingData.addSample(new double[]{1, 1}, new double[]{0});

        // Create validation data (same as training for this simple example)
        validationData = new DataSet(2, 1);
        validationData.addSample(new double[]{0, 0}, new double[]{0});
        validationData.addSample(new double[]{0, 1}, new double[]{1});
        validationData.addSample(new double[]{1, 0}, new double[]{1});
        validationData.addSample(new double[]{1, 1}, new double[]{0});

        // Create trainer
        trainer = new Trainer(network, trainingData, validationData, 4, 1000);
    }

    @Test
    void testTrainingProcess() {
        // Train the network
        trainer.train();

        // Test predictions
        double[] output1 = network.forward(new double[]{0, 0});
        double[] output2 = network.forward(new double[]{0, 1});
        double[] output3 = network.forward(new double[]{1, 0});
        double[] output4 = network.forward(new double[]{1, 1});

        // Check XOR logic
        assertTrue(output1[0] < 0.5);  // 0 XOR 0 = 0
        assertTrue(output2[0] > 0.5);  // 0 XOR 1 = 1
        assertTrue(output3[0] > 0.5);  // 1 XOR 0 = 1
        assertTrue(output4[0] < 0.5);  // 1 XOR 1 = 0
    }

    @Test
    void testTrainingProgress() {
        // Train the network
        trainer.train();

        // Check training history
        assertFalse(trainer.getTrainingHistory().isEmpty());
        assertFalse(trainer.getValidationHistory().isEmpty());

        // Check that loss decreases over time
        double[] trainingLosses = trainer.getTrainingHistory().stream()
            .mapToDouble(Double::doubleValue)
            .toArray();
        double[] validationLosses = trainer.getValidationHistory().stream()
            .mapToDouble(Double::doubleValue)
            .toArray();

        // Check that losses are decreasing
        for (int i = 1; i < trainingLosses.length; i++) {
            assertTrue(trainingLosses[i] <= trainingLosses[i-1]);
            assertTrue(validationLosses[i] <= validationLosses[i-1]);
        }
    }

    @Test
    void testModelSavingAndLoading() {
        // Train the network
        trainer.train();

        // Save the model
        String modelPath = "test_model.ser";
        ModelSerializer.saveModel(network, modelPath);

        // Load the model
        Network loadedNetwork = ModelSerializer.loadModel(modelPath);

        // Test predictions with loaded model
        double[] output1 = loadedNetwork.forward(new double[]{0, 0});
        double[] output2 = loadedNetwork.forward(new double[]{0, 1});
        double[] output3 = loadedNetwork.forward(new double[]{1, 0});
        double[] output4 = loadedNetwork.forward(new double[]{1, 1});

        // Check XOR logic with loaded model
        assertTrue(output1[0] < 0.5);  // 0 XOR 0 = 0
        assertTrue(output2[0] > 0.5);  // 0 XOR 1 = 1
        assertTrue(output3[0] > 0.5);  // 1 XOR 0 = 1
        assertTrue(output4[0] < 0.5);  // 1 XOR 1 = 0
    }

    @Test
    void testMiniBatchTraining() {
        // Create a larger dataset for mini-batch testing
        DataSet largeTrainingData = new DataSet(2, 1);
        for (int i = 0; i < 100; i++) {
            double x1 = Math.random() > 0.5 ? 1 : 0;
            double x2 = Math.random() > 0.5 ? 1 : 0;
            double target = (x1 == x2) ? 0 : 1;
            largeTrainingData.addSample(new double[]{x1, x2}, new double[]{target});
        }

        // Create trainer with smaller batch size
        Trainer miniBatchTrainer = new Trainer(network, largeTrainingData, validationData, 16, 10);

        // Train the network
        miniBatchTrainer.train();

        // Test predictions
        double[] output1 = network.forward(new double[]{0, 0});
        double[] output2 = network.forward(new double[]{0, 1});
        double[] output3 = network.forward(new double[]{1, 0});
        double[] output4 = network.forward(new double[]{1, 1});

        // Check XOR logic
        assertTrue(output1[0] < 0.5);  // 0 XOR 0 = 0
        assertTrue(output2[0] > 0.5);  // 0 XOR 1 = 1
        assertTrue(output3[0] > 0.5);  // 1 XOR 0 = 1
        assertTrue(output4[0] < 0.5);  // 1 XOR 1 = 0
    }
} 