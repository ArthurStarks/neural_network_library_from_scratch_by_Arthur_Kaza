# Neural Network Library

Une bibliothèque de réseaux de neurones implémentée à partir de zéro en Java, conçue pour l'apprentissage et l'expérimentation.

## Caractéristiques

- 🧠 Implémentation complète des réseaux de neurones
- 📊 Visualisation interactive des réseaux
- 🚀 Optimiseurs modernes (Adam, RMSprop, etc.)
- 🎯 Fonctions d'activation variées
- 📈 Suivi de l'entraînement en temps réel
- 💾 Sauvegarde et chargement de modèles

## Installation Rapide

```bash
git clone https://github.com/ArthurStarks/neural_network_library_from_scratch_by_Arthur_Kaza.git
cd neural_network_library_from_scratch_by_Arthur_Kaza
mvn clean install
```

## Exemple Simple

```java
Network network = new Network();
network.addLayer(new DenseLayer(784, 128));
network.addLayer(new ReLU());
network.addLayer(new DenseLayer(128, 10));
network.addLayer(new Softmax());

network.setOptimizer(new Adam(0.001));
network.train(trainingData, labels);
```

## Visualisation

La bibliothèque inclut des outils de visualisation puissants :

- Visualisation de la structure du réseau
- Graphiques d'entraînement en temps réel
- Exploration interactive
- Métriques détaillées

## Documentation

- [Guide d'Installation](installation.md)
- [Démarrage Rapide](quickstart.md)
- [Exemples de Code](examples.md)
- [Guide de Visualisation](visualization.md)

## Contribution

Nous accueillons les contributions ! Consultez notre [guide de contribution](CONTRIBUTING.md) pour commencer.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](../LICENSE) pour plus de détails. 