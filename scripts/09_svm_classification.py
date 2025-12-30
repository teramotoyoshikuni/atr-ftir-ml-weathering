# -*- coding: utf-8 -*-
"""
SVM classification (train/test split + CV for C and gamma)

Based on Dr. Kaneko's sample code.
- Input: for_classification.csv (index_col=0)
  Assumption: first column is class label, remaining columns are features.
- Output: confusion_matrix_train.csv, confusion_matrix_test.csv,
          estimated_y_train.csv, estimated_y_test.csv (overwritten if rerun)
"""

import numpy as np
import pandas as pd
import sample_functions
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

svm_cs = 2 ** np.arange(-5, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 11, dtype=float)

fold_number = 5
number_of_test_samples = 120

dataset = pd.read_csv('for_classification.csv', index_col=0)

# y = first column, x = rest
y_raw = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

# encode labels (works for string or numeric labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=number_of_test_samples, shuffle=True,
    random_state=99, stratify=y
)

# autoscaling
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# optimize gamma by maximizing variance of Gram matrix
optimal_svm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x_train, svm_gammas)

# CV optimize C
model_in_cv = GridSearchCV(
    svm.SVC(kernel='rbf', gamma=optimal_svm_gamma),
    {'C': svm_cs}, cv=fold_number
)
model_in_cv.fit(autoscaled_x_train, y_train)
optimal_svm_c = model_in_cv.best_params_['C']

# CV optimize gamma
model_in_cv = GridSearchCV(
    svm.SVC(kernel='rbf', C=optimal_svm_c),
    {'gamma': svm_gammas}, cv=fold_number
)
model_in_cv.fit(autoscaled_x_train, y_train)
optimal_svm_gamma = model_in_cv.best_params_['gamma']

print('CV optimized C :', optimal_svm_c)
print('CV optimized gamma :', optimal_svm_gamma)

# fit final model
model = svm.SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma, decision_function_shape='ovo')
model.fit(autoscaled_x_train, y_train)

# evaluate + save
sample_functions.estimation_and_performance_check_in_classification_train_and_test(
    model, autoscaled_x_train, y_train, autoscaled_x_test, y_test
)

# class index mapping
for class_index, class_label in enumerate(label_encoder.classes_):
    print(f'{class_index} : {class_label}')
