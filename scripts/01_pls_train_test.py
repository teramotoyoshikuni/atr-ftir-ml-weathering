# -*- coding: utf-8 -*-
"""
PLS regression with train/test split + CV to select n_components.

Input CSV (regression):
    data/for_regression.csv
    * 1st column: target variable (y)
    * 2nd and later columns: explanatory variables (X)

Notes:
- X for the test set is autoscaled using the training mean/std (recommended).
- y is autoscaled only for model training, and predictions are converted back to original scale.
- This script uses sample_functions.py without modification.

Outputs:
- outputs/pls_standard_regression_coefficients.csv
- outputs/estimated_y_train.csv
- outputs/estimated_y_test.csv
"""

from pathlib import Path
import pandas as pd
import sample_functions
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict

# ======================
# Settings
# ======================
DATA_PATH = "data/for_regression.csv"
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

max_number_of_principal_components = 20
fold_number = 5
number_of_test_samples = 20
RANDOM_STATE = 99

dataset = pd.read_csv(DATA_PATH, index_col=0)

# Split
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=number_of_test_samples, shuffle=True, random_state=RANDOM_STATE
)

# Autoscaling (IMPORTANT: test X must use train mean/std)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test  = (x_test  - x_train.mean()) / x_train.std()

# CV for selecting components
components = []
r2_in_cv_all = []

for component in range(1, max_number_of_principal_components + 1):
    model = PLSRegression(n_components=component)
    estimated_y_in_cv = pd.DataFrame(
        cross_val_predict(model, autoscaled_x_train, autoscaled_y_train, cv=fold_number)
    )
    estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()
    r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)
    print(component, r2_in_cv)

    r2_in_cv_all.append(r2_in_cv)
    components.append(component)

optimal_component_number = sample_functions.plot_and_selection_of_hyperparameter(
    components, r2_in_cv_all, "number of components", "cross-validated r2"
)
print("\nCV optimized n_components:", optimal_component_number)

# Train final model
model = PLSRegression(n_components=optimal_component_number)
model.fit(autoscaled_x_train, autoscaled_y_train)

# Standard regression coefficients
standard_regression_coefficients = pd.DataFrame(
    model.coef_.T, index=x_train.columns, columns=["standard_regression_coefficients"]
)
standard_regression_coefficients.to_csv(OUTDIR / "pls_standard_regression_coefficients.csv")

# Train/test evaluation + saving (handled inside sample_functions)
# NOTE: this function expects y_train/y_test in original scale and X autoscaled.
sample_functions.estimation_and_performance_check_in_regression_train_and_test(
    model, autoscaled_x_train, y_train, autoscaled_x_test, y_test
)

# Move outputs created by sample_functions into outputs/ (optional: do manually)
# sample_functions saves estimated_y_train.csv and estimated_y_test.csv to CWD by default.
