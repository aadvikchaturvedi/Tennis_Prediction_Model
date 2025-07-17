# 🎾 Tennis_Prediction_Model

This repository contains a machine learning model that predicts the outcome of professional tennis matches based on match statistics, player profiles, and historical performance.

---

## 📌 Project Overview

This project aims to predict match outcomes using engineered features such as:
- Player rank & point differences
- Serve statistics (aces, double faults, 1st/2nd serve win %)
- Head-to-head (H2H) and surface-specific H2H stats
- Break point saving efficiency
- Match metadata (surface, tournament level, draw size)

Model training and evaluation is done using **XGBoost Classifier**, with metrics like:
- Accuracy
- F1-score
- ROC AUC

---

## 📂 Files in the Repository

- `Tennis_Prediction_Model.ipynb` – Main Colab notebook containing all preprocessing, feature engineering, training, and evaluation.
- `data.csv` – Dataset used for training (if small enough to be uploaded).
- `README.md` – Project overview and setup guide.
- `requirements.txt` – Python dependencies (optional but recommended).

---

## 🛠️ Requirements

You can install dependencies using pip:

```bash
pip install -r requirements.txt

