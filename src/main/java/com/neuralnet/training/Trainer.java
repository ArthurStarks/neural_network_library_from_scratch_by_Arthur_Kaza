package com.neuralnet.training;

import com.neuralnet.core.Network;
import com.neuralnet.core.Loss;

import java.util.ArrayList;
import java.util.List;

public class Trainer {
    private final Network network;
    private final DataSet trainingData;
    private final DataSet validationData;
    private final DataPreprocessor preprocessor;
    private final int batchSize;
    private final int epochs;
    private final List<TrainingMetrics> trainingHistory;

    public Trainer(Network network, DataSet trainingData, DataSet validationData,
                  int batchSize, int epochs) {
        this.network = network;
        this.trainingData = trainingData;
        this.validationData = validationData;
        this.preprocessor = new DataPreprocessor(
            trainingData.getInputSize(),
            trainingData.getOutputSize()
        );
        this.batchSize = batchSize;
        this.epochs = epochs;
        this.trainingHistory = new ArrayList<>();
    }

    public void train() {
        // Fit preprocessor on training data
        preprocessor.fit(trainingData);

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Training phase
            double trainingLoss = trainEpoch();
            
            // Validation phase
            double validationLoss = validate();
            
            // Record metrics
            trainingHistory.add(new TrainingMetrics(epoch, trainingLoss, validationLoss));
            
            // Print progress
            System.out.printf("Epoch %d/%d - Training Loss: %.4f, Validation Loss: %.4f%n",
                epoch + 1, epochs, trainingLoss, validationLoss);
        }
    }

    private double trainEpoch() {
        double totalLoss = 0.0;
        int numBatches = 0;

        while (true) {
            DataSet.MiniBatch batch = trainingData.getMiniBatch(batchSize);
            if (batch == null) break;

            double[][] normalizedInputs = preprocessor.normalizeInputs(batch.getInputs());
            double[][] normalizedTargets = preprocessor.normalizeInputs(batch.getTargets());

            for (int i = 0; i < batch.size(); i++) {
                network.backward(normalizedInputs[i], normalizedTargets[i]);
                totalLoss += network.computeLoss(
                    network.forward(normalizedInputs[i]),
                    normalizedTargets[i]
                );
            }

            numBatches++;
        }

        return totalLoss / numBatches;
    }

    private double validate() {
        double totalLoss = 0.0;
        int numSamples = 0;

        for (int i = 0; i < validationData.size(); i++) {
            double[] normalizedInput = preprocessor.normalizeInput(
                validationData.getInputs().get(i)
            );
            double[] normalizedTarget = preprocessor.normalizeInput(
                validationData.getTargets().get(i)
            );

            double[] prediction = network.forward(normalizedInput);
            totalLoss += network.computeLoss(prediction, normalizedTarget);
            numSamples++;
        }

        return totalLoss / numSamples;
    }

    public List<TrainingMetrics> getTrainingHistory() {
        return new ArrayList<>(trainingHistory);
    }

    public static class TrainingMetrics {
        private final int epoch;
        private final double trainingLoss;
        private final double validationLoss;

        public TrainingMetrics(int epoch, double trainingLoss, double validationLoss) {
            this.epoch = epoch;
            this.trainingLoss = trainingLoss;
            this.validationLoss = validationLoss;
        }

        public int getEpoch() {
            return epoch;
        }

        public double getTrainingLoss() {
            return trainingLoss;
        }

        public double getValidationLoss() {
            return validationLoss;
        }
    }
} 