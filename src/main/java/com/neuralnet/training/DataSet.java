package com.neuralnet.training;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DataSet {
    private final List<double[]> inputs;
    private final List<double[]> targets;
    private final int inputSize;
    private final int outputSize;
    private final Random random;

    public DataSet(int inputSize, int outputSize) {
        this.inputs = new ArrayList<>();
        this.targets = new ArrayList<>();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.random = new Random();
    }

    public void addSample(double[] input, double[] target) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size must match dataset input size");
        }
        if (target.length != outputSize) {
            throw new IllegalArgumentException("Target size must match dataset output size");
        }
        inputs.add(input.clone());
        targets.add(target.clone());
    }

    public MiniBatch getMiniBatch(int batchSize) {
        if (batchSize > inputs.size()) {
            throw new IllegalArgumentException("Batch size cannot be larger than dataset size");
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            indices.add(i);
        }

        // Shuffle indices
        for (int i = indices.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices.get(i);
            indices.set(i, indices.get(j));
            indices.set(j, temp);
        }

        // Create mini-batch
        double[][] batchInputs = new double[batchSize][inputSize];
        double[][] batchTargets = new double[batchSize][outputSize];

        for (int i = 0; i < batchSize; i++) {
            int index = indices.get(i);
            System.arraycopy(inputs.get(index), 0, batchInputs[i], 0, inputSize);
            System.arraycopy(targets.get(index), 0, batchTargets[i], 0, outputSize);
        }

        return new MiniBatch(batchInputs, batchTargets);
    }

    public int size() {
        return inputs.size();
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public static class MiniBatch {
        private final double[][] inputs;
        private final double[][] targets;

        public MiniBatch(double[][] inputs, double[][] targets) {
            this.inputs = inputs;
            this.targets = targets;
        }

        public double[][] getInputs() {
            return inputs;
        }

        public double[][] getTargets() {
            return targets;
        }

        public int size() {
            return inputs.length;
        }
    }
} 