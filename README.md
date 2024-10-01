# FACEID-PYTHON

## Description

FACEID-PYTHON est un système de reconnaissance faciale complet utilisant les architectures FaceNet et VGG16. Ce projet vise à construire un pipeline de reconnaissance faciale, incluant la collecte de données, l'entraînement du modèle, les tests, ainsi que le déploiement avec CI/CD via GitHub Actions.

## Table des matières

1. [Caractéristiques](#caractéristiques)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Utilisation](#utilisation)
5. [Architecture du projet](#architecture-du-projet)
6. [Tests et validation](#tests-et-validation)
7. [Déploiement CI/CD](#déploiement-cicd)
8. [Contributeurs](#contributeurs)
9. [Licence](#licence)

## Caractéristiques

- **Modèles utilisés :** FaceNet et VGG16 pour la reconnaissance faciale.
- **Pipeline complet :** De la collecte de données à l’entraînement du modèle et au déploiement.
- **API RESTful :** Utilisation de Flask pour une interface permettant l'intégration du modèle en production.
- **CI/CD :** Automatisation des tests et du déploiement avec GitHub Actions.
- **Outils de traitement d'image :** OpenCV pour la détection des visages, augmentation des données pour améliorer les performances.

## Prérequis

Avant d'installer ce projet, assurez-vous d'avoir les outils suivants installés :

- **Python 3.8+**
- **TensorFlow 2.x**
- **Keras 2.x**
- **OpenCV**
- **Flask**
- **Git**
- **Docker** (facultatif, pour le déploiement)
- **GitHub Actions** (configuration pour CI/CD)

### Bibliothèques Python

Assurez-vous d'avoir installé les packages suivants :

```bash
pip install tensorflow keras opencv-python flask numpy pandas matplotlib scikit-learn

```

## Clonez le dépôt GitHub 
```bash
git clone https://github.com/astouthierno/FACEID-PYTHON.git
cd FACEID-PYTHON
```

## Installez les dépendances 

```bash
pip install -r requirements.txt
```

## Architecture du projet

FACEID-PYTHON/
│
├── data/                     # Dossier des données de visages
├── models/                   # Fichiers modèles sauvegardés (.h5)
├── src/
│   ├── train_model.py         # Script d'entraînement du modèle
│   ├── predict.py             # Script de prédiction
│   ├── app.py                 # API Flask pour la reconnaissance faciale
│   ├── webcam_recognition.py  # Script pour reconnaissance via webcam
│   └── utils/                 # Fonctions utilitaires
│
├── tests/                    # Scripts de tests
├── requirements.txt          # Fichier de dépendances Python
├── Dockerfile                # Fichier pour Docker
├── .github/                  # Configuration GitHub Actions pour CI/CD
│   └── workflows/            # Fichiers YAML pour CI/CD
│
└── README.md                 # Documentation du projet
