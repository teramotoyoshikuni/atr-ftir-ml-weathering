# -*- coding: utf-8 -*-
"""
kNN classification (train/test split + CV for k)

Based on Dr. Kaneko's sample code.
- Input: for_classification.csv (index_col=0)
- Output: confusion_matrix_train.csv, confusion_matrix_test.csv,
          estimated_y_train.csv, estimated_y_test.csv (overwritten if rerun)
"""

import pandas as pd
import sample_functions
from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

max_number_of_k = 20          # max k to search
fold_number = 5              # N-fold CV
number_of_test_samples = 120 # number of test samples

# load dataset
dataset = pd.read_csv('for_classification.csv', index_col=0)

# split x/y
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

# train/test split (reproducible)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=number_of_test_samples, shuffle=True,
    random_state=99, stratify=y
)

# autoscaling (fit on train, apply to test)
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# CV to optimize k
accuracy_in_cv_all = []
ks = []
for k in range(1, max_number_of_k + 1):
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    estimated_y_in_cv = pd.DataFrame(
        model_selection.cross_val_predict(model, autoscaled_x_train, y_train, cv=fold_number)
    )
    accuracy_in_cv = metrics.accuracy_score(y_train, estimated_y_in_cv)
    print(k, accuracy_in_cv)
    accuracy_in_cv_all.append(accuracy_in_cv)
    ks.append(k)

optimal_k = sample_functions.plot_and_selection_of_hyperparameter(
    ks, accuracy_in_cv_all, 'k', 'cross-validated accuracy'
)
print('\nCV optimized k :', optimal_k, '\n')

# fit final model
model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
model.fit(autoscaled_x_train, y_train)

# evaluate + save
sample_functions.estimation_and_performance_check_in_classification_train_and_test(
    model, autoscaled_x_train, y_train, autoscaled_x_test, y_test
)
