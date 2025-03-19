package com.neuralnet.training;

import java.util.Arrays;

public class DataPreprocessor {
    private final double[] inputMeans;
    private final double[] inputStdDevs;
    private final double[] targetMeans;
    private final double[] targetStdDevs;

    public DataPreprocessor(int inputSize, int outputSize) {
        this.inputMeans = new double[inputSize];
        this.inputStdDevs = new double[inputSize];
        this.targetMeans = new double[outputSize];
        this.targetStdDevs = new double[outputSize];
    }

    public void fit(DataSet dataset) {
        // Calculate means and standard deviations for inputs
        for (int i = 0; i < dataset.getInputSize(); i++) {
            double sum = 0.0;
            double sumSquared = 0.0;
            int count = 0;

            for (int j = 0; j < dataset.size(); j++) {
                double value = dataset.getInputs().get(j)[i];
                sum += value;
                sumSquared += value * value;
                count++;
            }

            inputMeans[i] = sum / count;
            inputStdDevs[i] = Math.sqrt((sumSquared / count) - (inputMeans[i] * inputMeans[i]));
            
            // Avoid division by zero
            if (inputStdDevs[i] == 0) {
                inputStdDevs[i] = 1.0;
            }
        }

        // Calculate means and standard deviations for targets
        for (int i = 0; i < dataset.getOutputSize(); i++) {
            double sum = 0.0;
            double sumSquared = 0.0;
            int count = 0;

            for (int j = 0; j < dataset.size(); j++) {
                double value = dataset.getTargets().get(j)[i];
                sum += value;
                sumSquared += value * value;
                count++;
            }

            targetMeans[i] = sum / count;
            targetStdDevs[i] = Math.sqrt((sumSquared / count) - (targetMeans[i] * targetMeans[i]));
            
            // Avoid division by zero
            if (targetStdDevs[i] == 0) {
                targetStdDevs[i] = 1.0;
            }
        }
    }

    public double[] normalizeInput(double[] input) {
        double[] normalized = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            normalized[i] = (input[i] - inputMeans[i]) / inputStdDevs[i];
        }
        return normalized;
    }

    public double[] denormalizeOutput(double[] output) {
        double[] denormalized = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            denormalized[i] = (output[i] * targetStdDevs[i]) + targetMeans[i];
        }
        return denormalized;
    }

    public double[][] normalizeInputs(double[][] inputs) {
        double[][] normalized = new double[inputs.length][inputs[0].length];
        for (int i = 0; i < inputs.length; i++) {
            normalized[i] = normalizeInput(inputs[i]);
        }
        return normalized;
    }

    public double[][] denormalizeOutputs(double[][] outputs) {
        double[][] denormalized = new double[outputs.length][outputs[0].length];
        for (int i = 0; i < outputs.length; i++) {
            denormalized[i] = denormalizeOutput(outputs[i]);
        }
        return denormalized;
    }

    public double[] getInputMeans() {
        return inputMeans.clone();
    }

    public double[] getInputStdDevs() {
        return inputStdDevs.clone();
    }

    public double[] getTargetMeans() {
        return targetMeans.clone();
    }

    public double[] getTargetStdDevs() {
        return targetStdDevs.clone();
    }
} 