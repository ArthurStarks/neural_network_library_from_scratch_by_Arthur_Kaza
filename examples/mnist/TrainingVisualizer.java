package com.neuralnetwork.examples.mnist;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.List;

public class TrainingVisualizer extends JFrame {
    private final List<Double> trainLosses = new ArrayList<>();
    private final List<Double> testLosses = new ArrayList<>();
    private final List<Double> accuracies = new ArrayList<>();
    private final JPanel graphPanel;
    private final JLabel statusLabel;
    private final int maxPoints = 100;

    public TrainingVisualizer() {
        setTitle("Neural Network Training Progress");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // Panel pour le graphique
        graphPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                drawGraph(g);
            }
        };
        graphPanel.setBackground(Color.WHITE);
        add(graphPanel, BorderLayout.CENTER);

        // Label pour le statut
        statusLabel = new JLabel("Initialisation...");
        statusLabel.setHorizontalAlignment(SwingConstants.CENTER);
        add(statusLabel, BorderLayout.SOUTH);

        // Centrer la fenêtre
        setLocationRelativeTo(null);
    }

    public void update(double trainLoss, double testLoss, double accuracy) {
        trainLosses.add(trainLoss);
        testLosses.add(testLoss);
        accuracies.add(accuracy);

        // Limiter le nombre de points affichés
        if (trainLosses.size() > maxPoints) {
            trainLosses.remove(0);
            testLosses.remove(0);
            accuracies.remove(0);
        }

        // Mettre à jour le statut
        statusLabel.setText(String.format("Train Loss: %.4f, Test Loss: %.4f, Accuracy: %.2f%%",
                trainLoss, testLoss, accuracy * 100));

        // Redessiner le graphique
        graphPanel.repaint();
    }

    private void drawGraph(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int width = graphPanel.getWidth();
        int height = graphPanel.getHeight();
        int padding = 50;

        // Dessiner les axes
        g2d.setColor(Color.BLACK);
        g2d.drawLine(padding, padding, padding, height - padding);
        g2d.drawLine(padding, height - padding, width - padding, height - padding);

        // Dessiner les courbes
        if (!trainLosses.isEmpty()) {
            // Courbe de perte d'entraînement (rouge)
            g2d.setColor(Color.RED);
            drawCurve(g2d, trainLosses, width, height, padding, 0, 1);

            // Courbe de perte de test (bleue)
            g2d.setColor(Color.BLUE);
            drawCurve(g2d, testLosses, width, height, padding, 0, 1);

            // Courbe de précision (verte)
            g2d.setColor(Color.GREEN);
            drawCurve(g2d, accuracies, width, height, padding, 0, 1);
        }
    }

    private void drawCurve(Graphics2D g2d, List<Double> values, int width, int height, int padding,
                          double minY, double maxY) {
        int points = values.size();
        if (points < 2) return;

        int[] xPoints = new int[points];
        int[] yPoints = new int[points];

        for (int i = 0; i < points; i++) {
            xPoints[i] = padding + (i * (width - 2 * padding)) / (points - 1);
            double normalizedY = (values.get(i) - minY) / (maxY - minY);
            yPoints[i] = height - padding - (int) (normalizedY * (height - 2 * padding));
        }

        g2d.drawPolyline(xPoints, yPoints, points);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            TrainingVisualizer visualizer = new TrainingVisualizer();
            visualizer.setVisible(true);

            // Simuler des données pour le test
            new Thread(() -> {
                for (int i = 0; i < 100; i++) {
                    try {
                        Thread.sleep(100);
                        double trainLoss = Math.random() * 0.5;
                        double testLoss = trainLoss + Math.random() * 0.1;
                        double accuracy = 0.5 + Math.random() * 0.5;
                        visualizer.update(trainLoss, testLoss, accuracy);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }).start();
        });
    }
} 