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


# Project 2 — Daily Demand Forecasting

## The Problem

Forecast daily demand for inventory and staffing planning.

## What I Did

- **Model:** XGBoost Regressor, treated as supervised regression on daily
  features
- **Tuning:** Optuna, 50 trials, optimizing **MAE** (not RMSE, to avoid
  letting a few extreme spikes dominate the objective)
- **Search space:** max_depth 3–6, learning_rate 0.01–0.10, n_estimators
  200–400

## Best Parameters

| Parameter | Value |
|---|---|
| max_depth | 4 |
| learning_rate | 0.0964 |
| n_estimators | 285 |

Best validation MAE: **18.3175**

## Test Results

| Metric | Value |
|---|---|
| MAE | 17.86 |
| RMSE | 55.37 |
| R² | 0.159 |

## What I Found Out

- **Generalizes well** — validation and test MAE nearly match, so no
  overfitting. The limit is signal, not tuning.
- **A few days dominate the error** — RMSE is ~3× MAE (vs. ~1.25× for
  normal errors), the signature of occasional large misses rather than
  uniform noise.
- **Low R² (0.159) follows from the same spikes** — the model has learned
  the baseline level of demand but not its volatility.
- **Depth 4 is a real sweet spot** — depth 3 underfit (MAE ~19.5), depth 5–6
  overfit (MAE ~20+), depth 4 alone reached 18.3–18.7.
- **Learning rate hit the ceiling again** (best trials all >0.09) — worth
  widening the search range on a rerun.

## Where the Improvements Are

- **Lag features** (t-1, t-7, t-14) — likely the biggest missing signal
- **Rolling mean/std** (7-day, 30-day) — captures recent volatility
- **Calendar/holiday flags** — explains spikes that look random now
- **Log-transform the target** — reduces the pull of extreme days
- **Wider learning-rate range** — model saturated at 0.10
  all 50 trials. The ceiling is the feature set, not the hyperparameters.
- **Learning rate hit the search ceiling** (0.08–0.10) — worth
