# Neural Network Library from Scratch

[![Java](https://img.shields.io/badge/Java-11-red.svg)](https://www.oracle.com/java/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/workflows/CI/badge.svg)](https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/actions)
[![Coverage Status](https://codecov.io/gh/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/branch/master/graph/badge.svg)](https://codecov.io/gh/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza)
[![Documentation Status](https://readthedocs.org/projects/neural-network-from-scratch/badge/?version=latest)](https://neural-network-from-scratch.readthedocs.io/en/latest/?badge=latest)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza/releases)

Une implémentation pédagogique d'une bibliothèque de réseaux de neurones en Java, conçue pour fournir une base théorique et pratique solide.

# Guide d'Installation

## Prérequis

- Java 11 ou supérieur
- Maven 3.6 ou supérieur
- Git

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza.git
cd neural_network_library_from_scratch_by_Arthur_Kaza
```

### 2. Compiler le projet

```bash
mvn clean install
```

### 3. Ajouter la dépendance à votre projet

Si vous utilisez Maven, ajoutez la dépendance suivante à votre `pom.xml` :

```xml
<dependency>
    <groupId>com.neuralnetwork</groupId>
    <artifactId>neural-network-library</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Vérification de l'installation

Pour vérifier que l'installation est correcte, vous pouvez exécuter les tests :

```bash
mvn test
```

## Configuration de l'environnement de développement

### 1. IDE recommandé

- IntelliJ IDEA
- Eclipse
- VS Code avec extensions Java

### 2. Extensions recommandées

- Lombok
- JUnit
- Maven

### 3. Configuration de la mémoire

Pour les grands réseaux de neurones, vous pouvez augmenter la mémoire JVM :

```bash
export MAVEN_OPTS="-Xmx4g"
```

## Dépannage

### Problèmes courants

1. **Erreur de compilation Java**
   - Vérifiez que Java 11 est installé : `java -version`
   - Vérifiez la variable d'environnement JAVA_HOME

2. **Erreur Maven**
   - Vérifiez que Maven est installé : `mvn -version`
   - Nettoyez le cache Maven : `mvn clean`

3. **Erreurs de dépendances**
   - Supprimez le dossier `.m2/repository/com/neuralnetwork`
   - Réexécutez `mvn clean install` 

## Fonctionnalités

### Couches de Base
- Dense (Fully Connected)
- Convolutionnelle
- Récurrente
- Pooling
- Normalisation par lots
- Dropout

### Optimiseurs
- SGD (Stochastic Gradient Descent)
- Adam
- RMSprop
- Adagrad

### Fonctions d'Activation
- ReLU
- Sigmoid
- Tanh
- Softmax
- Leaky ReLU
- ELU

### Fonctionnalités Avancées
- Backpropagation automatique
- Visualisation interactive
- Sauvegarde/chargement de modèles
- Traitement par mini-batches
- Prétraitement des données
- Suivi de la progression

## Installation

### Prérequis
- Java 11 ou supérieur
- Maven 3.6 ou supérieur

### Installation Directe
```bash
git clone https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza.git
cd neural_network_library_from_scratch_by_Arthur_Kaza
mvn clean install
```

### Installation via Maven
Ajoutez la dépendance suivante à votre `pom.xml` :
```xml
<dependency>
    <groupId>com.neuralnetwork</groupId>
    <artifactId>neural-network</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Exemples d'Utilisation

### Classification d'Images (MNIST)
```java
Network network = new Network();
network.addLayer(new ConvolutionalLayer(1, 32, 3, 3, 1, 1));
network.addLayer(new ReLU());
network.addLayer(new MaxPoolingLayer(2, 2, 2, 2));
network.addLayer(new ConvolutionalLayer(32, 64, 3, 3, 1, 1));
network.addLayer(new ReLU());
network.addLayer(new MaxPoolingLayer(2, 2, 2, 2));
network.addLayer(new FlattenLayer());
network.addLayer(new DenseLayer(64 * 7 * 7, 128));
network.addLayer(new ReLU());
network.addLayer(new DropoutLayer(0.5));
network.addLayer(new DenseLayer(128, 10));
network.addLayer(new Softmax());

network.setOptimizer(new Adam(0.001));
```

### Prédiction de Séries Temporelles
```java
Network network = new Network();
network.addLayer(new RecurrentLayer(inputSize, hiddenSize));
network.addLayer(new DenseLayer(hiddenSize, outputSize));
network.setOptimizer(new RMSprop(0.01));
```

### Classification de Texte
```java
Network network = new Network();
network.addLayer(new EmbeddingLayer(vocabularySize, embeddingSize));
network.addLayer(new LSTM(embeddingSize, hiddenSize));
network.addLayer(new DenseLayer(hiddenSize, numClasses));
network.setOptimizer(new Adam(0.001));
```

## Tests

### Compilation et Exécution des Tests
```bash
mvn test
```

### Couverture des Tests
```bash
mvn verify
```

## Documentation

La documentation complète est disponible dans le dossier `docs/` et inclut :
- Guide technique détaillé
- Documentation API
- FAQ
- Exemples de code
- Comparaisons de performance
- Guide de contribution

## Auteur

Arthur Kaza ([@ArthurStarks](https://github.com/ArthurStarks))

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails. 
