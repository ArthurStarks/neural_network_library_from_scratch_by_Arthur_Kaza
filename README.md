# Neural Network Library from Scratch

[![Java](https://img.shields.io/badge/Java-11%2B-red.svg)](https://www.oracle.com/java/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/actions)
[![Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen.svg)](https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/actions)
[![Documentation](https://img.shields.io/badge/Documentation-Complete-blue.svg)](docs/)
[![Version](https://img.shields.io/badge/Version-1.0.0-orange.svg)](https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/releases)

Une implémentation complète d'une bibliothèque de réseaux de neurones en Java, développée à partir de zéro par Arthur Kaza.

## Description

Ce projet présente une implémentation pédagogique d'une bibliothèque de réseaux de neurones en Java. L'objectif principal est de fournir une base théorique et pratique solide pour comprendre et implémenter les réseaux de neurones, tout en maintenant une architecture modulaire et extensible.

## Fonctionnalités

### Couches de Base
- **DenseLayer** : Couche entièrement connectée
- **ConvolutionLayer** : Couche de convolution pour le traitement d'images
- **RecurrentLayer** : Couche récurrente pour les séquences temporelles
- **PoolingLayer** : Couche de pooling pour la réduction de dimensionnalité
- **BatchNormalizationLayer** : Normalisation par lots pour un entraînement stable
- **DropoutLayer** : Régularisation par dropout

### Optimiseurs
- **SGD** : Descente de gradient stochastique
- **Adam** : Adaptive Moment Estimation
- **RMSprop** : Root Mean Square Propagation
- **Adagrad** : Adaptive Gradient Algorithm

### Fonctions d'Activation
- **ReLU** : Rectified Linear Unit
- **Sigmoid** : Fonction sigmoïde
- **Tanh** : Tangente hyperbolique
- **Softmax** : Normalisation exponentielle
- **LeakyReLU** : ReLU avec pente négative
- **ELU** : Exponential Linear Unit

### Fonctionnalités Avancées
- Rétropropagation automatique
- Visualisation interactive des réseaux
- Sauvegarde/chargement de modèles
- Traitement par lots (mini-batch)
- Prétraitement des données
- Suivi de la progression

## Installation

### Prérequis
- Java 11 ou supérieur
- Maven 3.6 ou supérieur (optionnel, pour la gestion des dépendances)

### Installation Directe
1. Clonez le dépôt :
```bash
git clone https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza.git
```

2. Compilez le projet :
```bash
javac src/main/java/com/neuralnetwork/*.java
```

### Installation avec Maven
1. Ajoutez le dépôt à votre `pom.xml` :
```xml
<repositories>
    <repository>
        <id>github</id>
        <url>https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/raw/master/maven-repo/</url>
    </repository>
</repositories>
```

2. Ajoutez la dépendance :
```xml
<dependency>
    <groupId>com.neuralnetwork</groupId>
    <artifactId>neural-network-library</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Exemples d'Utilisation

### Classification d'Images (MNIST)
```java
Network network = new Network();
network.addLayer(new ConvolutionLayer(1, 32, 3, 3, Activation.RELU));
network.addLayer(new MaxPoolingLayer(2, 2));
network.addLayer(new ConvolutionLayer(32, 64, 3, 3, Activation.RELU));
network.addLayer(new MaxPoolingLayer(2, 2));
network.addLayer(new DenseLayer(1600, 128, Activation.RELU));
network.addLayer(new DropoutLayer(0.5));
network.addLayer(new DenseLayer(128, 10, Activation.SOFTMAX));
network.setOptimizer(new AdamOptimizer(0.001));
network.train(mnistData, mnistLabels);
```

### Prédiction de Séries Temporelles
```java
Network network = new Network();
network.addLayer(new RecurrentLayer(1, 64, Activation.TANH));
network.addLayer(new DenseLayer(64, 1, Activation.LINEAR));
network.setOptimizer(new RMSpropOptimizer(0.01));
network.train(timeSeriesData, timeSeriesLabels);
```

### Classification de Texte
```java
Network network = new Network();
network.addLayer(new DenseLayer(vocabularySize, 256, Activation.RELU));
network.addLayer(new BatchNormalizationLayer());
network.addLayer(new DenseLayer(256, 128, Activation.RELU));
network.addLayer(new DropoutLayer(0.3));
network.addLayer(new DenseLayer(128, numClasses, Activation.SOFTMAX));
network.setOptimizer(new AdamOptimizer(0.001));
network.train(textData, textLabels);
```

## Tests

### Exécution des Tests
```bash
# Compilation des tests
javac -cp .:junit-4.13.2.jar:hamcrest-core-1.3.jar src/test/java/com/neuralnetwork/*.java

# Exécution des tests
java -cp .:junit-4.13.2.jar:hamcrest-core-1.3.jar org.junit.runner.JUnitCore com.neuralnetwork.LayerTest
```

### Couverture des Tests
- Tests unitaires pour toutes les couches
- Tests d'intégration pour les réseaux complets
- Tests de performance et de mémoire
- Tests de visualisation

## Documentation

La documentation complète est disponible dans le dossier `docs/`, incluant :
- [Guide Technique](docs/guide-technique.md)
- [API Documentation](docs/api-docs.md)
- [FAQ](docs/faq.md)
- [Exemples de Code](docs/examples.md)
- [Comparaison de Performance](docs/performance.md)
- [Guide de Contribution](docs/contributing.md)

## Structure du Projet

```
neural_network_library/
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── neuralnetwork/
│   │               ├── core/
│   │               ├── layers/
│   │               ├── optimizers/
│   │               ├── activations/
│   │               └── utils/
│   └── test/
│       └── java/
│           └── com/
│               └── neuralnetwork/
├── docs/
│   ├── guide-technique.md
│   ├── api-docs.md
│   ├── faq.md
│   └── examples.md
├── examples/
│   ├── mnist/
│   ├── time-series/
│   └── text-classification/
└── tests/
    ├── unit/
    └── integration/
```

## Auteur

- **Arthur Kaza** - [@ArthurStarks](https://github.com/ArthurStarks)
  - Développeur Full Stack
  - Spécialiste en Intelligence Artificielle
  - Passionné par l'enseignement et le partage de connaissances

## Contribution

Les contributions sont les bienvenues ! Veuillez consulter notre [guide de contribution](docs/contributing.md) pour plus de détails.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails. 