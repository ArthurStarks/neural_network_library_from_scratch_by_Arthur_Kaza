package com.neuralnetwork.examples.visualization;

import com.neuralnetwork.core.*;
import com.neuralnetwork.layers.*;
import com.neuralnetwork.optimizers.*;
import com.neuralnetwork.activations.*;

public class VisualizationExample {
    public static void main(String[] args) {
        // Créer le réseau de neurones
        Network network = new Network();
        network.addLayer(new DenseLayer(784, 128));
        network.addLayer(new ReLU());
        network.addLayer(new DenseLayer(128, 64));
        network.addLayer(new ReLU());
        network.addLayer(new DenseLayer(64, 10));
        network.addLayer(new Softmax());
        
        network.setOptimizer(new Adam(0.001));
        
        // Créer les visualiseurs
        NetworkVisualizer networkVisualizer = new NetworkVisualizer(network);
        TrainingVisualizer trainingVisualizer = new TrainingVisualizer();
        
        // Afficher les visualiseurs
        networkVisualizer.setVisible(true);
        trainingVisualizer.setVisible(true);
        
        // Simuler l'entraînement
        new Thread(() -> {
            for (int epoch = 0; epoch < 100; epoch++) {
                // Simuler une perte et une précision
                double loss = Math.exp(-epoch / 20.0) + Math.random() * 0.1;
                double accuracy = 1.0 - Math.exp(-epoch / 15.0) + Math.random() * 0.05;
                
                // Mettre à jour le visualiseur d'entraînement
                trainingVisualizer.update(loss, accuracy);
                
                // Mettre à jour le visualiseur du réseau
                networkVisualizer.repaint();
                
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
} 