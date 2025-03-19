package com.neuralnetwork.examples.visualization;

import com.neuralnetwork.core.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class NetworkVisualizer extends JFrame {
    private final Network network;
    private final JPanel networkPanel;
    private final JPanel controlPanel;
    private final Map<Layer, Color> layerColors;
    private double zoom = 1.0;
    private Point offset = new Point(0, 0);
    private Point lastMousePos;
    private boolean isDragging = false;

    public NetworkVisualizer(Network network) {
        this.network = network;
        this.layerColors = new HashMap<>();
        generateLayerColors();

        setTitle("Neural Network Visualizer");
        setSize(1200, 800);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // Panel pour le réseau
        networkPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                drawNetwork(g);
            }
        };
        networkPanel.setBackground(Color.WHITE);
        networkPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastMousePos = e.getPoint();
                isDragging = true;
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                isDragging = false;
            }
        });
        networkPanel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (isDragging) {
                    int dx = e.getX() - lastMousePos.x;
                    int dy = e.getY() - lastMousePos.y;
                    offset.translate(dx, dy);
                    lastMousePos = e.getPoint();
                    networkPanel.repaint();
                }
            }
        });
        networkPanel.addMouseWheelListener(e -> {
            double zoomFactor = e.getWheelRotation() < 0 ? 1.1 : 0.9;
            zoom *= zoomFactor;
            networkPanel.repaint();
        });
        add(networkPanel, BorderLayout.CENTER);

        // Panel de contrôle
        controlPanel = new JPanel();
        controlPanel.setLayout(new BoxLayout(controlPanel, BoxLayout.Y_AXIS));
        add(controlPanel, BorderLayout.EAST);

        // Boutons de contrôle
        JButton resetButton = new JButton("Reset View");
        resetButton.addActionListener(e -> {
            zoom = 1.0;
            offset.setLocation(0, 0);
            networkPanel.repaint();
        });
        controlPanel.add(resetButton);

        JButton infoButton = new JButton("Network Info");
        infoButton.addActionListener(e -> showNetworkInfo());
        controlPanel.add(infoButton);

        // Centrer la fenêtre
        setLocationRelativeTo(null);
    }

    private void generateLayerColors() {
        Random random = new Random();
        for (Layer layer : network.getLayers()) {
            layerColors.put(layer, new Color(
                random.nextInt(200) + 55,
                random.nextInt(200) + 55,
                random.nextInt(200) + 55
            ));
        }
    }

    private void drawNetwork(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Appliquer la transformation
        g2d.translate(offset.x, offset.y);
        g2d.scale(zoom, zoom);

        // Dimensions du panel
        int width = networkPanel.getWidth();
        int height = networkPanel.getHeight();

        // Dessiner les couches
        List<Layer> layers = network.getLayers();
        int layerWidth = width / (layers.size() + 1);
        int layerHeight = height - 100;

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            int x = (i + 1) * layerWidth;
            int y = 50;

            // Dessiner la couche
            g2d.setColor(layerColors.get(layer));
            g2d.fillRect(x - 40, y, 80, layerHeight);

            // Dessiner les connexions
            if (i > 0) {
                Layer prevLayer = layers.get(i - 1);
                drawConnections(g2d, prevLayer, layer, x - layerWidth, x, y, layerHeight);
            }

            // Dessiner le titre de la couche
            g2d.setColor(Color.BLACK);
            g2d.drawString(layer.getClass().getSimpleName(), x - 40, y - 10);
        }
    }

    private void drawConnections(Graphics2D g2d, Layer from, Layer to,
                               int fromX, int toX, int y, int height) {
        g2d.setColor(new Color(200, 200, 200, 100));
        int numFrom = from.getOutputSize();
        int numTo = to.getInputSize();

        for (int i = 0; i < numFrom; i++) {
            for (int j = 0; j < numTo; j++) {
                int fromY = y + (i * height) / (numFrom - 1);
                int toY = y + (j * height) / (numTo - 1);
                g2d.drawLine(fromX + 40, fromY, toX - 40, toY);
            }
        }
    }

    private void showNetworkInfo() {
        StringBuilder info = new StringBuilder();
        info.append("Network Information:\n\n");
        
        List<Layer> layers = network.getLayers();
        info.append("Total Layers: ").append(layers.size()).append("\n\n");
        
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            info.append("Layer ").append(i + 1).append(":\n");
            info.append("  Type: ").append(layer.getClass().getSimpleName()).append("\n");
            info.append("  Input Size: ").append(layer.getInputSize()).append("\n");
            info.append("  Output Size: ").append(layer.getOutputSize()).append("\n");
            info.append("  Parameters: ").append(layer.getParameters().size()).append("\n\n");
        }

        JTextArea textArea = new JTextArea(info.toString());
        textArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(textArea);
        scrollPane.setPreferredSize(new Dimension(400, 300));

        JDialog dialog = new JDialog(this, "Network Information", true);
        dialog.add(scrollPane);
        dialog.pack();
        dialog.setLocationRelativeTo(this);
        dialog.setVisible(true);
    }

    public static void main(String[] args) {
        // Créer un réseau de test
        Network network = new Network();
        network.addLayer(new DenseLayer(784, 128));
        network.addLayer(new ReLU());
        network.addLayer(new DenseLayer(128, 64));
        network.addLayer(new ReLU());
        network.addLayer(new DenseLayer(64, 10));
        network.addLayer(new Softmax());

        // Afficher le visualiseur
        SwingUtilities.invokeLater(() -> {
            NetworkVisualizer visualizer = new NetworkVisualizer(network);
            visualizer.setVisible(true);
        });
    }
} 