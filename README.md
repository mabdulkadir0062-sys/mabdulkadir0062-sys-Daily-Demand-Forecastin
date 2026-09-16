# Machine Learning Projects — Customer Churn & Daily Demand Forecasting

Two end-to-end supervised learning projects built with XGBoost and
hyperparameter-tuned with Optuna. One is a classification problem with
imbalanced classes; the other is a regression problem with heavy-tailed
errors. Both are documented below with the full tuning process, the final
metrics, and an honest read of what the numbers mean.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Optuna 

---

# Project 1 — Customer Churn Prediction

## The Problem

Predict which customers are likely to churn, so a retention team can
prioritize outreach. Test set is imbalanced (1,303 no-churn vs 697 churn,
~65/35), so accuracy alone would be misleading.

## What I Did

- **Model:** XGBoost Classifier
- **Imbalance handling:** tuned `scale_pos_weight` as a search parameter
  instead of fixing it at the class ratio
- **Tuning:** Optuna, 50 trials, optimizing **F1 on the churn class** (not
  accuracy, which would reward predicting "no churn" too often)
- **Search space:** max_depth 3–6, learning_rate 0.01–0.10, n_estimators
  200–400, scale_pos_weight 1.0–2.8

## Best Parameters

| Parameter | Value |
|---|---|
| max_depth | 3 |
| learning_rate | 0.0979 |
| n_estimators | 288 |
| scale_pos_weight | 2.145 |

Best validation F1: **0.6578**

## Test Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No churn | 0.85 | 0.69 | 0.76 | 1,303 |
| Churn | 0.57 | 0.77 | 0.65 | 697 |
| **Accuracy** | | | **0.72** | 2,000 |

## What I Found Out

- **Recall over precision, by design** — catches 77% of churners but flags
  some non-churners too (57% precision). Right trade-off when a missed
  churner costs more than a wasted retention offer.
- **Class weighting mattered most** — the one low-weight trial
  (`scale_pos_weight` ≈ 1.1) gave the worst F1 (0.591); every top trial used
  1.9–2.8.
- **Shallow trees won** — depth 3–4 outperformed depth 5–6 consistently,
  meaning churn is driven by a few strong signals, not deep interactions.
- **Tuning had limited room to help** — F1 only ranged 0.591–0.658 across
  all 50 trials. The ceiling is the feature set, not the hyperparameters.
- **Learning rate hit the search ceiling** (0.08–0.10) — worth
