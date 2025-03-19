package com.neuralnetwork.examples.mnist;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MNISTDataLoader {
    private static final String TRAINING_IMAGES = "train-images-idx3-ubyte";
    private static final String TRAINING_LABELS = "train-labels-idx1-ubyte";
    private static final String TEST_IMAGES = "t10k-images-idx3-ubyte";
    private static final String TEST_LABELS = "t10k-labels-idx1-ubyte";
    private static final String DATA_DIR = "data/mnist/";

    public double[][][] loadTrainingData() throws IOException {
        return loadImages(TRAINING_IMAGES);
    }

    public double[][] loadTrainingLabels() throws IOException {
        return loadLabels(TRAINING_LABELS);
    }

    public double[][][] loadTestData() throws IOException {
        return loadImages(TEST_IMAGES);
    }

    public double[][] loadTestLabels() throws IOException {
        return loadLabels(TEST_LABELS);
    }

    private double[][][] loadImages(String filename) throws IOException {
        String path = DATA_DIR + filename;
        FileInputStream fis = new FileInputStream(path);
        DataInputStream dis = new DataInputStream(fis);

        // Lire l'en-tête
        int magicNumber = dis.readInt();
        int numberOfImages = dis.readInt();
        int numberOfRows = dis.readInt();
        int numberOfColumns = dis.readInt();

        // Créer le tableau de données
        double[][][] data = new double[numberOfImages][numberOfRows][numberOfColumns];

        // Lire les images
        for (int i = 0; i < numberOfImages; i++) {
            for (int r = 0; r < numberOfRows; r++) {
                for (int c = 0; c < numberOfColumns; c++) {
                    // Normaliser les valeurs entre 0 et 1
                    data[i][r][c] = dis.readUnsignedByte() / 255.0;
                }
            }
        }

        dis.close();
        fis.close();
        return data;
    }

    private double[][] loadLabels(String filename) throws IOException {
        String path = DATA_DIR + filename;
        FileInputStream fis = new FileInputStream(path);
        DataInputStream dis = new DataInputStream(fis);

        // Lire l'en-tête
        int magicNumber = dis.readInt();
        int numberOfLabels = dis.readInt();

        // Créer le tableau de labels (one-hot encoding)
        double[][] labels = new double[numberOfLabels][10];

        // Lire les labels
        for (int i = 0; i < numberOfLabels; i++) {
            int label = dis.readUnsignedByte();
            labels[i][label] = 1.0; // One-hot encoding
        }

        dis.close();
        fis.close();
        return labels;
    }

    public void downloadMNISTData() throws IOException {
        // Créer le répertoire de données s'il n'existe pas
        File dataDir = new File(DATA_DIR);
        if (!dataDir.exists()) {
            dataDir.mkdirs();
        }

        // URLs des fichiers MNIST
        String baseUrl = "http://yann.lecun.com/exdb/mnist/";
        String[] files = {
            TRAINING_IMAGES,
            TRAINING_LABELS,
            TEST_IMAGES,
            TEST_LABELS
        };

        // Télécharger chaque fichier
        for (String file : files) {
            String url = baseUrl + file;
            String path = DATA_DIR + file;
            
            // Vérifier si le fichier existe déjà
            if (!new File(path).exists()) {
                System.out.println("Téléchargement de " + file + "...");
                downloadFile(url, path);
            }
        }
    }

    private void downloadFile(String url, String path) throws IOException {
        try (java.io.InputStream in = new java.net.URL(url).openStream();
             java.io.FileOutputStream out = new java.io.FileOutputStream(path)) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
    }
} 