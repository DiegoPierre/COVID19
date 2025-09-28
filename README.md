# 🐊 Analyse des espèces mondiales de crocodiles

## Aperçu du projet

Ce projet utilise un ensemble de données sur les crocodiles du monde entier afin d'analyser leurs caractéristiques biologiques et géographiques.  
L'objectif est de comprendre la distribution des espèces, de prédire certaines caractéristiques comme la taille adulte ou le poids, et d'identifier les facteurs les plus importants pour la conservation et la gestion des populations.  

Le dataset contient des informations sur :  
- Le type et la classification de l'espèce  
- Les caractéristiques biologiques (taille, poids, âge à maturité)  
- La répartition géographique  
- Les observations enregistrées dans le monde entier

Ce projet permettra de construire des modèles de régression et de classification, d'évaluer leur performance et de fournir des recommandations pratiques pour la conservation des crocodiles.


###  Compréhension du métier (Business Understanding)

**Problématique principale :**  
Comment prédire certaines caractéristiques des crocodiles (taille, poids, classification, etc.) et identifier les facteurs influençant leur distribution dans le monde, afin de soutenir la recherche et la conservation des espèces ?

**Parties prenantes :**  
- Chercheurs et biologistes spécialisés en crocodiles  
- Organisations de protection de la faune  
- Data scientists et analystes de données  

**Objectifs du projet :**  
- Construire des modèles supervisés pour prédire la classification ou des caractéristiques biologiques des crocodiles (régression ou classification).  
- Identifier les facteurs influençant la taille, le poids et la répartition des espèces.  
- Fournir des recommandations exploitables pour la recherche et la conservation.

### Data Understanding

L'objectif de cette étape est de comprendre la structure et le contenu du dataset avant de commencer la préparation et la modélisation.

L'ensemble de données `crocodile_dataset.csv` contient des informations sur plusieurs espèces de crocodiles dans le monde. Chaque enregistrement inclut :  
- Le type et la classification de l'espèce (genre, espèce, famille)  
- Les caractéristiques biologiques (taille adulte, poids, âge à maturité, etc.)  
- La répartition géographique et habitats naturels  
- Les observations et mesures enregistrées par les chercheurs  
- Les comportements et interactions avec l’environnement  
- Les éventuelles notes ou commentaires des chercheurs  

**Objectifs de l'exploration des données :**  
- Identifier les colonnes **numériques** et **catégorielles**  
- Détecter les **valeurs manquantes ou aberrantes**  
- Comprendre la **distribution des espèces, tailles et poids**  
- Fournir un **premier aperçu des relations entre les variables**, qui guidera la préparation des données et le choix des modèles  

Cette étape permettra de préparer un **jeu de données propre et exploitable** pour la modélisation supervisée (régression ou classification) et pour des analyses exploratoires comme le clustering ou la visualisation géographique.


###  Data Preparation

Cette étape vise à préparer le dataset pour la modélisation supervisée et non supervisée.  

**Étapes réalisées :**  
1. **Nettoyage des données**  
   - Suppression des doublons pour éviter les biais dans les modèles.  

2. **Gestion des valeurs manquantes**  
   - Colonnes numériques : remplissage avec la médiane.  
   - Colonnes catégorielles : remplissage avec la valeur la plus fréquente.  

3. **Transformation des variables**  
   - Conversion des colonnes de type texte ou date en formats exploitables (ex. année de naissance, âge, etc.)  
   - Création de nouvelles features si nécessaire (ex. ratio poids/taille, âge relatif).  

4. **Encodage des variables catégorielles**  
   - Transformation des colonnes telles que `Genre`, `Famille`, `Habitat` en valeurs numériques à l’aide de `LabelEncoder` ou `OneHotEncoder`.  

5. **Mise à l’échelle des features numériques**  
   - Standardisation des colonnes comme `Taille adulte`, `Poids`, `Âge à maturité` pour les modèles sensibles à l’échelle (KNN, réseaux neuronaux).  

6. **Séparation des données**  
   - Création des jeux `X_train`, `X_test`, `y_train`, `y_test` selon la variable cible choisie (ex. classification de l’espèce ou prédiction du poids).  

Cette préparation garantit que les modèles de classification, régression, clustering et analyses avancées puissent être appliqués efficacement et produire des résultats fiables et interprétables.

# 📌 Résumé du code d'importation des bibliothèques

Le code commence par importer toutes les bibliothèques nécessaires pour :

---

## 🔹 1. Manipuler les données
- **pandas** : pour charger, transformer et analyser les données tabulaires.  
- **numpy** : pour les calculs numériques et la manipulation de tableaux.

---

## 🔹 2. Visualiser les résultats
- **matplotlib** : pour créer des graphiques simples (courbes, barres, histogrammes).  
- **seaborn** : pour des visualisations plus esthétiques et avancées (heatmaps, boxplots, corrélations).

---

## 🔹 3. Préparer et entraîner des modèles de Machine Learning
- **train_test_split** : séparer le dataset en un ensemble d’entraînement et un ensemble de test.  
- **LabelEncoder** : transformer les variables catégorielles en valeurs numériques.  
- **StandardScaler** : normaliser les données pour rendre les variables comparables.  

- **RandomForestClassifier / Regressor** : modèles d’arbres de décision robustes pour classification et régression.  
- **KNeighborsClassifier / Regressor** : algorithme basé sur la proximité des points (k plus proches voisins).  

- **GridSearchCV / RandomizedSearchCV** : optimisation des hyperparamètres des modèles.

---

## 🔹 4. Évaluer les performances des modèles
- **Classification** :  
  - accuracy_score  
  - precision_score  
  - recall_score  
  - f1_score  
  - confusion_matrix  
  - classification_report  

- **Régression** :  
  - mean_squared_error (MSE)  
  - r2_score (coefficient de détermination)

``` python
# =========================
# Import des librairies
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error, r2_score
```


# 📌 Lecture du dataset

Le code permet de charger et d’avoir un premier aperçu des données.

---

## 🔹 1. Chargement du dataset
```python
# =========================
data = pd.read_csv("crocodile_dataset.csv")

# Affichage des premières lignes
data.head(10)

```

# 🧹 Nettoyage des doublons

Le but est de vérifier si certaines lignes du dataset sont présentes plusieurs fois et, si c’est le cas, de les supprimer afin d’éviter les biais dans l’analyse.

---

## 🔹 1. Vérification du nombre de doublons
```python
# Nettoyage des doublons
# =========================
print("Nombre de doublons :", data.duplicated().sum())

# Suppression des doublons
data = data.drop_duplicates()

# Vérification
print("Nouvelle dimension du dataset :", data.shape)

```

# 🧩 Gestion des valeurs manquantes

L’objectif est de traiter les valeurs manquantes (`NaN`) dans le dataset afin de garantir la qualité des analyses et d’éviter les erreurs lors de l’entraînement des modèles.

---

## 🔹 1. Identification des colonnes numériques
```python
# =========================
#Gestion des valeurs manquantes
# =========================

# Colonnes numériques
numeric_cols = data.select_dtypes(include=['float64','int64']).columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# Colonnes catégorielles
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Vérification
print("Valeurs manquantes par colonne après traitement :")
print(data.isnull().sum())

```

