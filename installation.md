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