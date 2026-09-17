# Daily Demand Forecasting

Forecasting daily demand to support inventory and staffing decisions.
Built with XGBoost and hyperparameter-tuned with Optuna.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Optuna

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
