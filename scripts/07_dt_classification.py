# -*- coding: utf-8 -*-
"""
Decision Tree classification (train/test split + CV for max_depth)

Based on Dr. Kaneko's sample code.
- Input: for_classification.csv (index_col=0)
- Output: confusion_matrix_train.csv, confusion_matrix_test.csv,
          estimated_y_train.csv, estimated_y_test.csv, tree.dot
          (overwritten if rerun)
"""

import pandas as pd
import sample_functions
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_predict

max_max_depth = 10       # upper limit (searched as 1..max_max_depth-1)
min_samples_leaf = 3
fold_number = 5
number_of_test_samples = 120

dataset = pd.read_csv('for_classification.csv', index_col=0)

y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=number_of_test_samples, shuffle=True,
    random_state=99, stratify=y
)

# CV to optimize max_depth
accuracy_cv = []
max_depthes = []
for max_depth in range(1, max_max_depth):
    model_in_cv = tree.DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf
    )
    estimated_y_in_cv = cross_val_predict(model_in_cv, x_train, y_train, cv=fold_number)
    accuracy_cv.append(metrics.accuracy_score(y_train, estimated_y_in_cv))
    max_depthes.append(max_depth)

optimal_max_depth = sample_functions.plot_and_selection_of_hyperparameter(
    max_depthes, accuracy_cv, 'maximum depth of tree', 'accuracy in CV'
)
print('\nCV optimized max_depth :', optimal_max_depth)

# fit final model
model = tree.DecisionTreeClassifier(
    max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf
)
model.fit(x_train, y_train)

# evaluate + save
sample_functions.estimation_and_performance_check_in_classification_train_and_test(
    model, x_train, y_train, x_test, y_test
)

# export DOT
with open('tree.dot', 'w') as f:
    if model.classes_.dtype == 'object':
        class_names = model.classes_
    else:
        class_names = [str(c) for c in model.classes_]
    tree.export_graphviz(model, out_file=f, feature_names=x.columns, class_names=class_names)
