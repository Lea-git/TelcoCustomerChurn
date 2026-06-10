DRUFFIN Léa
# Prédiction et Analyse de Données avec Machine Learning et Deep Learning

## 📖 Description

Ce projet a pour objectif d'explorer différentes approches de Machine Learning et de Deep Learning pour la prédiction à partir d'un jeu de données. Il comprend une phase d'expérimentation dans des notebooks Jupyter ainsi qu'une application web permettant d'utiliser le modèle entraîné.

---

## 📂 Structure du projet

```text
project/
│
├── notebooks/
│   ├── note1.ipynb
│   ├── note2.ipynb
│   └── note3.ipynb
│
├── models/
│   ├── encoder.pkl
│   ├── baseline.pkl
│   ├── elasticnet.pkl
│   ├── lasso.pkl
│   ├── random_forest.pkl
│   ├── ridge.pkl
│   ├── ridge2.pkl
│   ├── scaler.pkl
│   └── mlp.pkl
│
├── dashboard.py
├── requirements.txt
└── README.md
```

### Dossier `notebooks`

Le dossier `notebooks` contient trois notebooks :

* **Notebook 1** : expérimentations et prétraitement des données.
* **Notebook 2** : tests et comparaison de plusieurs algorithmes de Machine Learning.
* **Notebook 3** : implémentation et évaluation d'un modèle de Deep Learning.

### Dossier `models`

Contient les objets sauvegardés après entraînement :

* `encoder.pkl` : encodeur utilisé pour transformer les variables catégorielles.
* `scaler.pkl` : normalisation des données numériques.
* `mlp.pkl` : modèle de réseau de neurones entraîné.

### Application Web

Le fichier `dashboard.py` constitue le point d'entrée de l'application et joue le rôle de `app.py`. Il permet d'interagir avec le modèle via une interface utilisateur.

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <url-du-repository>
cd <nom-du-projet>
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
```

Activation :

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## ▶️ Lancer l'application

```bash
python dashboard.py
```

Ou selon le framework utilisé :

### Streamlit

```bash
streamlit run dashboard.py
```

### Dash

```bash
python dashboard.py
```

L'application sera accessible depuis votre navigateur.

---

## 🧠 Modèles utilisés

Le projet explore plusieurs approches :

### Machine Learning

* Régression logistique
* Arbres de décision
* Random Forest
* Support Vector Machine (SVM)
* Réseau de neurones MLP

### Deep Learning

* Réseaux de neurones profonds
* Évaluation et comparaison avec les modèles classiques

---

## 📊 Workflow

1. Chargement des données
2. Nettoyage et prétraitement
3. Encodage des variables catégorielles
4. Normalisation des données
5. Entraînement des modèles
6. Évaluation des performances
7. Sauvegarde du modèle
8. Déploiement dans l'application web

---

## 💾 Sauvegarde des modèles

Les modèles sont sauvegardés avec Joblib :

```python
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(mlp, "models/mlp.pkl")
```

Chargement :

```python
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
mlp = joblib.load("models/mlp.pkl")
```

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre d'un apprentissage et d'une expérimentation autour du Machine Learning et du Deep Learning.

Dataset : Telco Customer Churn

Téléchargement :
https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data

Après téléchargement, placer le fichier :

data/raw/dataset.csv
