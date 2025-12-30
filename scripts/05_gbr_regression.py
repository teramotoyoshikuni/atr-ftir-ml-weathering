# -*- coding: utf-8 -*-
"""
Gradient Boosting Regression (GBR) with Bayesian Optimization.

- Input CSV (regression):
    data/for_regression.csv
    * 1st column: target variable (y)
    * 2nd and later columns: explanatory variables (X)

- Hyperparameters are optimized using Bayesian Optimization on CV R2
  within the training set (to avoid test leakage).
- Final evaluation is performed once on the held-out test set.
- Outputs are saved in 'outputs/'.

This script is provided for research and educational purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics

from bayes_opt import BayesianOptimization

# ======================
# Settings
# ======================
DATA_PATH = "data/for_regression.csv"
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

RANDOM_STATE = 99
N_TEST_SAMPLES = 20

CV_FOLDS = 5  # CV folds used inside training set for Bayesian optimization

# ======================
# Load dataset
# ======================
dataset = pd.read_csv(DATA_PATH, index_col=0)

y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

# ======================
# Train / test split
# ======================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=N_TEST_SAMPLES, shuffle=True, random_state=RANDOM_STATE
)

# ======================
# Objective function for Bayesian optimization
# (optimize CV R2 on training set only)
# ======================
cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def gbr_objective(n_estimators, learning_rate, max_depth):
    model = GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        random_state=RANDOM_STATE
    )
    # Use mean CV R2 as objective
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="r2")
    return float(scores.mean())

pbounds = {
    "n_estimators": (10, 500),
    "learning_rate": (0.01, 0.5),
    "max_depth": (2, 20),
}

optimizer = BayesianOptimization(
    f=gbr_objective,
    pbounds=pbounds,
    random_state=RANDOM_STATE,
    verbose=2
)

print("\n--- Bayesian optimization (CV R2 on training set) ---")
optimizer.maximize(init_points=5, n_iter=20)

best_params = optimizer.max["params"]
print("\nBest params:", best_params)
print("Best CV R2:", optimizer.max["target"])

# ======================
# Train final model with best params
# ======================
final_gbr_model = GradientBoostingRegressor(
    n_estimators=int(best_params["n_estimators"]),
    learning_rate=float(best_params["learning_rate"]),
    max_depth=int(best_params["max_depth"]),
    random_state=RANDOM_STATE
)
final_gbr_model.fit(x_train, y_train)

# ======================
# Train predictions
# ======================
estimated_y_train = pd.DataFrame(
    final_gbr_model.predict(x_train),
    index=x_train.index,
    columns=["estimated y"]
)

plt.rcParams["font.size"] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, estimated_y_train.iloc[:, 0], c="blue")
y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())
y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], "k-")
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel("actual y")
plt.ylabel("estimated y")
plt.show()

print("r^2 for training data :", metrics.r2_score(y_train, estimated_y_train))
print("RMSE for training data :", metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
print("MAE for training data :", metrics.mean_absolute_error(y_train, estimated_y_train))

results_train = pd.concat([
    estimated_y_train,
    y_train.rename("actual y"),
    (y_train - estimated_y_train.iloc[:, 0]).rename("error (actual - estimated)")
], axis=1)
results_train.to_csv(OUTDIR / "gbr_estimated_y_train.csv")

# ======================
# Test predictions (final, only once)
# ======================
estimated_y_test = pd.DataFrame(
    final_gbr_model.predict(x_test),
    index=x_test.index,
    columns=["estimated y"]
)

plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_test, estimated_y_test.iloc[:, 0], c="blue")
y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())
y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], "k-")
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel("actual y")
plt.ylabel("estimated y")
plt.show()

print("r^2 for test data :", metrics.r2_score(y_test, estimated_y_test))
print("RMSE for test data :", metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5)
print("MAE for test data :", metrics.mean_absolute_error(y_test, estimated_y_test))

results_test = pd.concat([
    estimated_y_test,
    y_test.rename("actual y"),
    (y_test - estimated_y_test.iloc[:, 0]).rename("error (actual - estimated)")
], axis=1)
results_test.to_csv(OUTDIR / "gbr_estimated_y_test.csv")

# Save best params
pd.Series(best_params).to_csv(OUTDIR / "gbr_best_params.csv")
