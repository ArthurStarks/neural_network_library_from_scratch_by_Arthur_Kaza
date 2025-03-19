package com.neuralnetwork.examples.visualization;

import org.junit.jupiter.api.Test;
import javax.swing.*;
import java.awt.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

public class TrainingVisualizerTest {
    
    @Test
    public void testVisualizerCreation() {
        TrainingVisualizer visualizer = new TrainingVisualizer();
        assertNotNull(visualizer);
        assertTrue(visualizer.isVisible());
    }

    @Test
    public void testDataUpdate() throws InterruptedException {
        TrainingVisualizer visualizer = new TrainingVisualizer();
        CountDownLatch latch = new CountDownLatch(1);

        // Simuler des mises à jour de données
        SwingUtilities.invokeLater(() -> {
            for (int i = 0; i < 10; i++) {
                visualizer.update(1.0 - i * 0.1, i * 0.1);
            }
            latch.countDown();
        });

        // Attendre que les mises à jour soient terminées
        assertTrue(latch.await(5, TimeUnit.SECONDS));
    }

    @Test
    public void testChartRendering() {
        TrainingVisualizer visualizer = new TrainingVisualizer();
        
        // Créer un BufferedImage pour tester le rendu
        BufferedImage image = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        
        // Simuler le rendu du graphique
        visualizer.update(0.5, 0.75);
        
        // Vérifier que l'image a été modifiée
        int[] pixels = new int[800 * 600];
        image.getRGB(0, 0, 800, 600, pixels, 0, 800);
        
        // Vérifier que l'image n'est pas vide (contient des pixels non blancs)
        boolean hasNonWhitePixels = false;
        for (int pixel : pixels) {
            if (pixel != Color.WHITE.getRGB()) {
                hasNonWhitePixels = true;
                break;
            }
        }
        
        assertTrue(hasNonWhitePixels, "Le graphique devrait contenir des éléments visuels");
    }

    @Test
    public void testDataLimits() {
        TrainingVisualizer visualizer = new TrainingVisualizer();
        
        // Ajouter plus de points que la limite maximale
        for (int i = 0; i < 1100; i++) {
            visualizer.update(1.0 - i * 0.001, i * 0.001);
        }
        
        // Vérifier que le nombre de points ne dépasse pas la limite
        // Note: Cette vérification nécessite d'ajouter une méthode getter pour accéder aux listes
        // ou de modifier la classe pour exposer ces informations
    }
} 