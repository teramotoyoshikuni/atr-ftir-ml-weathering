# -*- coding: utf-8 -*-
"""
Random Forest classification (train/test split + OOB for max_features)

Based on Dr. Kaneko's sample code.
- Input: for_classification.csv (index_col=0)
- Output: rf_x_importances.csv, confusion_matrix_train.csv, confusion_matrix_test.csv,
          estimated_y_train.csv, estimated_y_test.csv (overwritten if rerun)
"""

import math
import numpy as np
import pandas as pd
import sample_functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rf_number_of_trees = 300
rf_x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 0.1..1.0

number_of_test_samples = 120
dataset = pd.read_csv('for_classification.csv', index_col=0)

y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=number_of_test_samples, shuffle=True,
    random_state=99, stratify=y
)

# OOB to optimize max_features rate
accuracy_oob = []
for index, x_variables_rate in enumerate(rf_x_variables_rates):
    print(index + 1, '/', len(rf_x_variables_rates))
    model_in_validation = RandomForestClassifier(
        n_estimators=rf_number_of_trees,
        max_features=int(max(math.ceil(x_train.shape[1] * x_variables_rate), 1)),
        oob_score=True
    )
    model_in_validation.fit(x_train, y_train)
    accuracy_oob.append(model_in_validation.oob_score_)

optimal_x_variables_rate = sample_functions.plot_and_selection_of_hyperparameter(
    list(rf_x_variables_rates), accuracy_oob, 'rate of x-variables', 'accuracy for OOB'
)
print('\nOOB optimized rate of x-variables :', optimal_x_variables_rate)

# fit final model
model = RandomForestClassifier(
    n_estimators=rf_number_of_trees,
    max_features=int(max(math.ceil(x_train.shape[1] * optimal_x_variables_rate), 1)),
    oob_score=True
)
model.fit(x_train, y_train)

# feature importances
x_importances = pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=['importance'])
x_importances.to_csv('rf_x_importances.csv')

# evaluate + save
sample_functions.estimation_and_performance_check_in_classification_train_and_test(
    model, x_train, y_train, x_test, y_test
)
