# Exemples de Code

Cette page contient des exemples détaillés d'utilisation de la bibliothèque.

## Classification d'Images (MNIST)

```java
public class MNISTExample {
    public static void main(String[] args) {
        // Chargement des données
        MNISTDataLoader loader = new MNISTDataLoader();
        double[][][] trainingData = loader.loadTrainingData();
        int[] trainingLabels = loader.loadTrainingLabels();
        
        // Création du réseau
        Network network = new Network();
        network.addLayer(new ConvolutionalLayer(1, 32, 3, 3));
        network.addLayer(new MaxPoolingLayer(2, 2));
        network.addLayer(new FlattenLayer());
        network.addLayer(new DenseLayer(1568, 128));
        network.addLayer(new ReLU());
        network.addLayer(new Dropout(0.5));
        network.addLayer(new DenseLayer(128, 10));
        network.addLayer(new Softmax());
        
        // Configuration
        network.setOptimizer(new Adam(0.001));
        network.setBatchSize(32);
        
        // Visualisation
        NetworkVisualizer networkVisualizer = new NetworkVisualizer(network);
        TrainingVisualizer trainingVisualizer = new TrainingVisualizer();
        networkVisualizer.setVisible(true);
        trainingVisualizer.setVisible(true);
        
        // Entraînement
        for (int epoch = 0; epoch < 10; epoch++) {
            double loss = network.train(trainingData, trainingLabels);
            double accuracy = network.evaluate(testData, testLabels);
            
            trainingVisualizer.update(loss, accuracy);
            networkVisualizer.repaint();
            
            System.out.printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%%n", 
                            epoch + 1, loss, accuracy * 100);
        }
        
        // Sauvegarde du modèle
        network.save("mnist_model.ser");
    }
}
```

## Prédiction de Séries Temporelles

```java
public class TimeSeriesExample {
    public static void main(String[] args) {
        // Préparation des données
        double[][] data = loadTimeSeriesData();
        double[][] X = prepareSequences(data, sequenceLength);
        double[][] y = prepareTargets(data, sequenceLength);
        
        // Création du réseau
        Network network = new Network();
        network.addLayer(new RecurrentLayer(inputSize, hiddenSize));
        network.addLayer(new DenseLayer(hiddenSize, outputSize));
        
        // Configuration
        network.setOptimizer(new RMSprop(0.01));
        network.setBatchSize(16);
        
        // Entraînement
        for (int epoch = 0; epoch < 100; epoch++) {
            double loss = network.train(X, y);
            System.out.printf("Epoch %d: Loss = %.4f%n", epoch + 1, loss);
        }
        
        // Prédiction
        double[] prediction = network.predict(nextSequence);
        System.out.println("Prédiction: " + prediction[0]);
    }
}
```

## Classification de Texte

```java
public class TextClassificationExample {
    public static void main(String[] args) {
        // Préparation des données
        Tokenizer tokenizer = new Tokenizer();
        double[][] embeddings = loadWordEmbeddings();
        int[][] sequences = tokenizer.tokenize(texts);
        
        // Création du réseau
        Network network = new Network();
        network.addLayer(new EmbeddingLayer(vocabularySize, embeddingSize, embeddings));
        network.addLayer(new LSTM(hiddenSize));
        network.addLayer(new DenseLayer(hiddenSize, numClasses));
        network.addLayer(new Softmax());
        
        // Configuration
        network.setOptimizer(new Adam(0.001));
        network.setBatchSize(32);
        
        // Entraînement
        for (int epoch = 0; epoch < 50; epoch++) {
            double loss = network.train(sequences, labels);
            double accuracy = network.evaluate(testSequences, testLabels);
            
            System.out.printf("Epoch %d: Loss = %.4f, Accuracy = %.2f%%%n", 
                            epoch + 1, loss, accuracy * 100);
        }
    }
}
```

## Visualisation Avancée

```java
public class VisualizationExample {
    public static void main(String[] args) {
        Network network = createNetwork();
        
        // Création des visualiseurs
        NetworkVisualizer networkVisualizer = new NetworkVisualizer(network);
        TrainingVisualizer trainingVisualizer = new TrainingVisualizer();
        
        // Configuration des visualiseurs
        networkVisualizer.setTitle("Structure du Réseau");
        networkVisualizer.setShowWeights(true);
        networkVisualizer.setShowBiases(true);
        
        trainingVisualizer.setTitle("Progression de l'Entraînement");
        trainingVisualizer.setShowGrid(true);
        trainingVisualizer.setAutoScale(true);
        
        // Affichage
        networkVisualizer.setVisible(true);
        trainingVisualizer.setVisible(true);
        
        // Mise à jour en temps réel
        while (training) {
            trainingVisualizer.update(loss, accuracy);
            networkVisualizer.repaint();
            Thread.sleep(100);
        }
    }
}
```

## Sauvegarde et Chargement

```java
public class ModelPersistenceExample {
    public static void main(String[] args) {
        // Sauvegarde
        Network network = createAndTrainNetwork();
        network.save("model.ser");
        
        // Chargement
        Network loadedNetwork = Network.load("model.ser");
        
        // Vérification
        double[] originalOutput = network.predict(input);
        double[] loadedOutput = loadedNetwork.predict(input);
        
        System.out.println("Différence: " + 
            Math.abs(originalOutput[0] - loadedOutput[0]));
    }
} 