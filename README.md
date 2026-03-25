---

# Système d’Irrigation Intelligent et Autonome

## Description

Ce projet vise à développer un système d’irrigation intelligent, autonome et économe en eau, spécialement conçu pour les **zones désertiques** (ex. la région d’El Oued en Algérie). Grâce à l'intégration de **modèles prédictifs LSTM** et d’un **agent d’apprentissage par renforcement profond (DQN)**, le système adapte en temps réel l’irrigation en fonction des besoins des cultures (tomate) et des conditions climatiques locales.

## Objectifs du Projet

- Optimiser la gestion de l’eau dans des zones à stress hydrique élevé.
- Améliorer les rendements agricoles en personnalisant l’irrigation.
- Réduire les coûts d’irrigation par une gestion intelligente des ressources.
- Intégrer des technologies d’**IA**, de simulation **agro-climatique**, et de **capteurs IoT**.

## Technologies et Méthodologies Utilisées

- **Deep Learning** : LSTM pour la prédiction du SWTD (soil water content) et du rendement.
- **Reinforcement Learning** : Deep Q-Network (DQN) pour la décision optimale d’irrigation.
- **Simulation agronomique** : DSSAT avec le module CROPGRO-Tomato (via DSSATTools).
- **Sources de données** : NASA POWER pour les données climatiques.
- **Prétraitement** : Z-Score, normalisation, PCA (selon les notebooks).
- **Évaluation** : RMSE, R² pour la précision des modèles.

## Modèles

### LSTM 1 : Prédiction SWTD

- Entrées : variables climatiques et irrigation (cf. `docs/data.md`).
- Fenêtre temporelle : 4 à 7 jours.
- Sortie : teneur en eau du sol le lendemain.

### LSTM 2 : Prédiction du rendement

- Entrées : climat, irrigation historique, SWTD (saison complète).
- Sortie : estimation du rendement/biomasse en fin de saison (proxy via `CWAD` selon DSSAT).

### Agent DQN : Optimisation des volumes d’irrigation

- Actions possibles : 12 niveaux (0–60 mm/jour).
- Récompense : rendement économique − coût de l’eau.

## Spécificités Régionales

- **Localisation** : Wilaya d’El Oued, climat saharien extrême.
- **Culture ciblée** : Tomate (variétés locales adaptées aux fortes chaleurs).
- **Irrigation** : pivots agricoles mobiles automatisés.

## Auteurs

- **Yazi Lynda Mellissa**
- **Benmachiche Khaled**

---
