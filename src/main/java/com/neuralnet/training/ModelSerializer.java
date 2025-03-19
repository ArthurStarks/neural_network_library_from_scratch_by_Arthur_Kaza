package com.neuralnet.training;

import com.neuralnet.core.*;
import com.neuralnet.optimizers.Optimizer;
import java.io.*;
import java.util.List;

public class ModelSerializer {
    public static void saveModel(Network network, String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(filePath))) {
            oos.writeObject(network);
        }
    }

    public static Network loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(filePath))) {
            return (Network) ois.readObject();
        }
    }

    public static void saveTrainingState(Network network, DataPreprocessor preprocessor,
                                      String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(filePath))) {
            oos.writeObject(network);
            oos.writeObject(preprocessor);
        }
    }

    public static class TrainingState {
        private final Network network;
        private final DataPreprocessor preprocessor;

        public TrainingState(Network network, DataPreprocessor preprocessor) {
            this.network = network;
            this.preprocessor = preprocessor;
        }

        public Network getNetwork() {
            return network;
        }

        public DataPreprocessor getPreprocessor() {
            return preprocessor;
        }
    }

    public static TrainingState loadTrainingState(String filePath) 
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileOutputStream(filePath))) {
            Network network = (Network) ois.readObject();
            DataPreprocessor preprocessor = (DataPreprocessor) ois.readObject();
            return new TrainingState(network, preprocessor);
        }
    }
} 