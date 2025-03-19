package com.neuralnetwork.examples.visualization;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class TrainingVisualizer extends JFrame {
    private final List<Double> losses;
    private final List<Double> accuracies;
    private final JPanel chartPanel;
    private final JLabel lossLabel;
    private final JLabel accuracyLabel;
    private final int maxPoints = 1000;
    private double minLoss = Double.MAX_VALUE;
    private double maxLoss = Double.MIN_VALUE;
    private double minAccuracy = 1.0;
    private double maxAccuracy = 0.0;

    public TrainingVisualizer() {
        losses = new ArrayList<>();
        accuracies = new ArrayList<>();

        setTitle("Training Progress");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // Panel pour le graphique
        chartPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                drawChart(g);
            }
        };
        chartPanel.setBackground(Color.WHITE);
        add(chartPanel, BorderLayout.CENTER);

        // Panel pour les statistiques
        JPanel statsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        lossLabel = new JLabel("Loss: N/A");
        accuracyLabel = new JLabel("Accuracy: N/A");
        statsPanel.add(lossLabel);
        statsPanel.add(accuracyLabel);
        add(statsPanel, BorderLayout.SOUTH);

        // Centrer la fenêtre
        setLocationRelativeTo(null);
    }

    public void update(double loss, double accuracy) {
        losses.add(loss);
        accuracies.add(accuracy);

        // Mettre à jour les min/max
        minLoss = Math.min(minLoss, loss);
        maxLoss = Math.max(maxLoss, loss);
        minAccuracy = Math.min(minAccuracy, accuracy);
        maxAccuracy = Math.max(maxAccuracy, accuracy);

        // Limiter le nombre de points
        if (losses.size() > maxPoints) {
            losses.remove(0);
            accuracies.remove(0);
        }

        // Mettre à jour les labels
        lossLabel.setText(String.format("Loss: %.4f", loss));
        accuracyLabel.setText(String.format("Accuracy: %.2f%%", accuracy * 100));

        // Redessiner le graphique
        chartPanel.repaint();
    }

    private void drawChart(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int width = chartPanel.getWidth();
        int height = chartPanel.getHeight();
        int padding = 50;

        // Dessiner les axes
        g2d.setColor(Color.BLACK);
        g2d.drawLine(padding, padding, padding, height - padding);
        g2d.drawLine(padding, height - padding, width - padding, height - padding);

        // Dessiner les courbes
        if (!losses.isEmpty()) {
            // Courbe de perte
            g2d.setColor(Color.RED);
            drawCurve(g2d, losses, minLoss, maxLoss, padding, width, height, true);

            // Courbe d'accuracy
            g2d.setColor(Color.BLUE);
            drawCurve(g2d, accuracies, minAccuracy, maxAccuracy, padding, width, height, false);
        }

        // Dessiner les légendes
        g2d.setColor(Color.RED);
        g2d.drawString("Loss", width - 100, padding + 20);
        g2d.setColor(Color.BLUE);
        g2d.drawString("Accuracy", width - 100, padding + 40);
    }

    private void drawCurve(Graphics2D g2d, List<Double> values, double min, double max,
                          int padding, int width, int height, boolean isLoss) {
        int points = values.size();
        if (points < 2) return;

        int[] xPoints = new int[points];
        int[] yPoints = new int[points];

        for (int i = 0; i < points; i++) {
            xPoints[i] = padding + (i * (width - 2 * padding)) / (points - 1);
            double value = values.get(i);
            int y;
            if (isLoss) {
                y = height - padding - (int) ((value - min) * (height - 2 * padding) / (max - min));
            } else {
                y = height - padding - (int) ((value - min) * (height - 2 * padding) / (max - min));
            }
            yPoints[i] = y;
        }

        g2d.drawPolyline(xPoints, yPoints, points);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            TrainingVisualizer visualizer = new TrainingVisualizer();
            visualizer.setVisible(true);

            // Simulation de données d'entraînement
            new Thread(() -> {
                for (int i = 0; i < 100; i++) {
                    double loss = Math.exp(-i / 20.0) + Math.random() * 0.1;
                    double accuracy = 1.0 - Math.exp(-i / 15.0) + Math.random() * 0.05;
                    visualizer.update(loss, accuracy);
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }).start();
        });
    }
} 