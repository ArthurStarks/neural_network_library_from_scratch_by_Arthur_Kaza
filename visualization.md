# Visualisation des Réseaux de Neurones

Ce document décrit les fonctionnalités de visualisation disponibles dans la bibliothèque de réseaux de neurones.

## Types de Visualisation

### 1. Visualiseur de Réseau (NetworkVisualizer)

Le `NetworkVisualizer` permet de visualiser la structure d'un réseau de neurones de manière interactive.

#### Fonctionnalités
- Affichage de la structure du réseau en temps réel
- Zoom et déplacement interactifs
- Affichage des informations détaillées sur les couches
- Mise à jour dynamique pendant l'entraînement

#### Utilisation
```java
Network network = new Network();
// ... configuration du réseau ...
NetworkVisualizer visualizer = new NetworkVisualizer(network);
visualizer.setVisible(true);
```

### 2. Visualiseur d'Entraînement (TrainingVisualizer)

Le `TrainingVisualizer` affiche les métriques d'entraînement en temps réel.

#### Fonctionnalités
- Graphique de la perte (loss) en temps réel
- Graphique de la précision (accuracy) en temps réel
- Mise à jour automatique des courbes
- Limitation du nombre de points affichés pour les performances

#### Utilisation
```java
TrainingVisualizer visualizer = new TrainingVisualizer();
// Pendant l'entraînement
visualizer.update(loss, accuracy);
```

## Exemple Complet

Voici un exemple complet d'utilisation des visualiseurs :

```java
public class VisualizationExample {
    public static void main(String[] args) {
        // Création du réseau
        Network network = new Network();
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
        
        // Boucle d'entraînement
        for (int epoch = 0; epoch < 100; epoch++) {
            // Entraînement...
            double loss = network.train(batch);
            double accuracy = network.evaluate(testData);
            
            // Mise à jour des visualiseurs
            trainingVisualizer.update(loss, accuracy);
            networkVisualizer.repaint();
        }
    }
}
```

## Tests d'Intégration

Les visualiseurs sont accompagnés de tests d'intégration complets :

### Tests du NetworkVisualizer
- Test de création et d'affichage
- Test des interactions utilisateur
- Test de la mise à jour dynamique

### Tests du TrainingVisualizer
- Test de création et d'affichage
- Test de mise à jour des données
- Test du rendu du graphique
- Test des limites de données

## Bonnes Pratiques

1. **Performance**
   - Limiter la fréquence de mise à jour des visualiseurs
   - Utiliser des threads séparés pour l'entraînement et la visualisation
   - Gérer correctement la fermeture des fenêtres

2. **Interface Utilisateur**
   - Placer les visualiseurs dans des fenêtres séparées
   - Utiliser des titres descriptifs
   - Ajouter des légendes aux graphiques

3. **Gestion des Erreurs**
   - Vérifier la validité des données avant la visualisation
   - Gérer les cas limites (données manquantes, valeurs aberrantes)
   - Afficher des messages d'erreur appropriés

## Limitations

- Les visualiseurs sont optimisés pour des réseaux de taille moyenne
- La performance peut être affectée par un grand nombre de points de données
- L'interface graphique nécessite un environnement avec support Swing 