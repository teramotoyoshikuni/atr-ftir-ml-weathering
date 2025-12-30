# -*- coding: utf-8 -*-
"""
PLS regression with LOOCV (Leave-One-Out Cross-Validation).

Input CSV (regression):
    data/for_regression.csv
    * 1st column: target variable (y)
    * 2nd and later columns: explanatory variables (X)

Notes:
- Autoscaling is performed *within each LOOCV fold* using the training mean/std
  (to avoid data leakage).
- This script uses sample_functions.py without modification.

Outputs:
- pls_standard_regression_coefficients.csv
- estimated_y_loocv.csv
"""

import pandas as pd
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
import sample_functions
import matplotlib.pyplot as plt

max_number_of_principal_components = 20
dataset = pd.read_csv("data/for_regression.csv", index_col=0)

y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

loo = LeaveOneOut()

# ----------------------
# LOOCV for selecting n_components
# ----------------------
components = []
r2_in_loocv_all = []

for component in range(1, max_number_of_principal_components + 1):
    estimated_y_in_loocv = []

    for train_index, test_index in loo.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # autoscale using TRAIN statistics
        x_mean, x_std = x_train.mean(), x_train.std()
        y_mean, y_std = y_train.mean(), y_train.std()

        autoscaled_x_train = (x_train - x_mean) / x_std
        autoscaled_y_train = (y_train - y_mean) / y_std
        autoscaled_x_test = (x_test - x_mean) / x_std

        model = PLSRegression(n_components=component)
        model.fit(autoscaled_x_train, autoscaled_y_train)

        y_pred = model.predict(autoscaled_x_test)[0, 0]  # shape: (1,1)
        estimated_y_in_loocv.append(y_pred * y_std + y_mean)

    r2_in_loocv = metrics.r2_score(y, estimated_y_in_loocv)
    print(f"n_components={component}, LOOCV R2={r2_in_loocv}")
    r2_in_loocv_all.append(r2_in_loocv)
    components.append(component)

optimal_component_number = sample_functions.plot_and_selection_of_hyperparameter(
    components, r2_in_loocv_all, "number of components", "LOOCV R2"
)
print("\nOptimal n_components (LOOCV):", optimal_component_number)

# ----------------------
# Fit final model on ALL data (for coefficients only)
# ----------------------
autoscaled_x_all = (x - x.mean()) / x.std()
autoscaled_y_all = (y - y.mean()) / y.std()

final_model = PLSRegression(n_components=optimal_component_number)
final_model.fit(autoscaled_x_all, autoscaled_y_all)

standard_regression_coefficients = pd.DataFrame(
    final_model.coef_.T,
    index=x.columns,
    columns=["standard_regression_coefficients"],
)
standard_regression_coefficients.to_csv("pls_standard_regression_coefficients.csv")

# ----------------------
# LOOCV predictions with the selected n_components
# ----------------------
estimated_y_loocv = []

for train_index, test_index in loo.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    autoscaled_x_train = (x_train - x_mean) / x_std
    autoscaled_y_train = (y_train - y_mean) / y_std
    autoscaled_x_test = (x_test - x_mean) / x_std

    model = PLSRegression(n_components=optimal_component_number)
    model.fit(autoscaled_x_train, autoscaled_y_train)

    y_pred = model.predict(autoscaled_x_test)[0, 0]
    estimated_y_loocv.append(y_pred * y_std + y_mean)

loocv_r2 = metrics.r2_score(y, estimated_y_loocv)
loocv_rmse = metrics.mean_squared_error(y, estimated_y_loocv, squared=False)
loocv_mae = metrics.mean_absolute_error(y, estimated_y_loocv)

print(f"LOOCV R2: {loocv_r2}")
print(f"LOOCV RMSE: {loocv_rmse}")
print(f"LOOCV MAE: {loocv_mae}")

plt.figure(figsize=(8, 8))
plt.scatter(y, estimated_y_loocv)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k-")
plt.xlabel("Actual y")
plt.ylabel("Estimated y")
plt.title("LOOCV Actual vs. Estimated y")
plt.show()

results_loocv = pd.DataFrame(
    {"Actual y": y, "Estimated y": estimated_y_loocv, "Error (Actual y - Estimated y)": y - estimated_y_loocv}
)
results_loocv.to_csv("estimated_y_loocv.csv", index=False)
