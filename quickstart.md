# Démarrage Rapide

Ce guide vous aidera à démarrer rapidement avec la bibliothèque de réseaux de neurones.

## Création d'un réseau simple

```java
import com.neuralnetwork.core.*;
import com.neuralnetwork.layers.*;
import com.neuralnetwork.optimizers.*;
import com.neuralnetwork.activations.*;

public class QuickStartExample {
    public static void main(String[] args) {
        // Création du réseau
        Network network = new Network();
        
        // Ajout des couches
        network.addLayer(new DenseLayer(784, 128));
        network.addLayer(new ReLU());
        network.addLayer(new DenseLayer(128, 64));
        network.addLayer(new ReLU());
        network.addLayer(new DenseLayer(64, 10));
        network.addLayer(new Softmax());
        
        // Configuration de l'optimiseur
        network.setOptimizer(new Adam(0.001));
        
        // Création des visualiseurs
        NetworkVisualizer networkVisualizer = new NetworkVisualizer(network);
        TrainingVisualizer trainingVisualizer = new TrainingVisualizer();
        
        // Affichage des visualiseurs
        networkVisualizer.setVisible(true);
        trainingVisualizer.setVisible(true);
        
        // Entraînement
        for (int epoch = 0; epoch < 100; epoch++) {
            double loss = network.train(batch);
            double accuracy = network.evaluate(testData);
            
            // Mise à jour des visualiseurs
            trainingVisualizer.update(loss, accuracy);
            networkVisualizer.repaint();
        }
    }
}
```

## Fonctionnalités principales

### 1. Création de couches

```java
// Couche dense
DenseLayer dense = new DenseLayer(inputSize, outputSize);

// Couche de convolution
ConvolutionalLayer conv = new ConvolutionalLayer(inputChannels, outputChannels, kernelSize, kernelSize);

// Couche récurrente
RecurrentLayer recurrent = new RecurrentLayer(inputSize, hiddenSize);

// Couche de pooling
MaxPoolingLayer pooling = new MaxPoolingLayer(poolSize, poolSize);
```

### 2. Fonctions d'activation

```java
// ReLU
network.addLayer(new ReLU());

// Sigmoid
network.addLayer(new Sigmoid());

// Tanh
network.addLayer(new Tanh());

// Softmax
network.addLayer(new Softmax());
```

### 3. Optimiseurs

```java
// SGD
network.setOptimizer(new SGD(0.01));

// Adam
network.setOptimizer(new Adam(0.001));

// RMSprop
network.setOptimizer(new RMSprop(0.01));

// Adagrad
network.setOptimizer(new Adagrad(0.01));
```

## Visualisation

### 1. Structure du réseau

```java
NetworkVisualizer visualizer = new NetworkVisualizer(network);
visualizer.setVisible(true);
```

### 2. Progression de l'entraînement

```java
TrainingVisualizer visualizer = new TrainingVisualizer();
visualizer.update(loss, accuracy);
```

## Sauvegarde et chargement

### 1. Sauvegarder un modèle

```java
network.save("model.ser");
```

### 2. Charger un modèle

```java
Network network = Network.load("model.ser");
```

## Prochaines étapes

- Consultez la [documentation complète](index.md)
- Explorez les [exemples](examples.md)
- Apprenez à [contribuer](CONTRIBUTING.md) 