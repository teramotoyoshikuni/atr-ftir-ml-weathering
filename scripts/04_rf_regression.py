# -*- coding: utf-8 -*-
"""
Random Forest regression for FTIR-based prediction.

- Input CSV (regression):
    data/for_regression.csv
    * 1st column: target variable (y)
    * 2nd and later columns: explanatory variables (X)

- Random seeds are fixed for reproducibility.
- Outputs are saved in the 'outputs/' directory.

This script is provided for research and educational purposes.
"""

import pandas as pd
import numpy as np
import matplotlib.figure as figure
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# ======================
# Settings
# ======================
DATA_PATH = "data/for_regression.csv"
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

RANDOM_STATE = 99
N_ESTIMATORS = 500

# ======================
# Load dataset
# ======================
dataset = pd.read_csv(DATA_PATH, index_col=0)

x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# ======================
# Train / test split
# ======================
# test_size=20 means "20 samples" (not 20%)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=20, shuffle=True, random_state=RANDOM_STATE
)

# ======================
# OOB-based optimization of max_features
# ======================
ratios_of_x = np.arange(0.1, 1.1, 0.1)
r2_oob = []

for ratio_of_x in ratios_of_x:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=ratio_of_x,
        oob_score=True,
        random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    r2_oob.append(model.oob_score_)

plt.rcParams['font.size'] = 18
plt.scatter(ratios_of_x, r2_oob)
plt.xlabel('ratio of x')
plt.ylabel('r2 for OOB')
plt.show()

optimal_ratio_of_x = ratios_of_x[r2_oob.index(max(r2_oob))]

# ======================
# Train final RF model
# ======================
model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_features=optimal_ratio_of_x,
    oob_score=True,
    random_state=RANDOM_STATE
)
model.fit(x_train, y_train)

# ======================
# Feature importance
# ======================
importances = pd.DataFrame(
    model.feature_importances_,
    index=x_train.columns,
    columns=['importance']
)
importances.to_csv(OUTDIR / "rf_feature_importances.csv")

# ======================
# Training data results
# ======================
estimated_y_train = pd.DataFrame(
    model.predict(x_train),
    index=x_train.index,
    columns=['estimated y']
)

results_train = pd.concat([
    estimated_y_train,
    y_train.rename('actual y'),
    (y_train - estimated_y_train.iloc[:, 0]).rename('error (actual - estimated)')
], axis=1)

results_train.to_csv(OUTDIR / "rf_estimated_y_train.csv")

print('Training R2:', metrics.r2_score(y_train, estimated_y_train))
print('Training RMSE:', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
print('Training MAE:', metrics.mean_absolute_error(y_train, estimated_y_train))

# ======================
# Test data results
# ======================
estimated_y_test = pd.DataFrame(
    model.predict(x_test),
    index=x_test.index,
    columns=['estimated y']
)

results_test = pd.concat([
    estimated_y_test,
    y_test.rename('actual y'),
    (y_test - estimated_y_test.iloc[:, 0]).rename('error (actual - estimated)')
], axis=1)

results_test.to_csv(OUTDIR / "rf_estimated_y_test.csv")

print('Test R2:', metrics.r2_score(y_test, estimated_y_test))
print('Test RMSE:', metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5)
print('Test MAE:', metrics.mean_absolute_error(y_test, estimated_y_test))
