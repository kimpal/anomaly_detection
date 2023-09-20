#!/usr/bin/env python
# coding: utf-8

# nearness imports
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys

train_dataset = "./train_pp3_multi.csv"
test_dataset =  "./test_pp3_multi.csv"
def DF_XY_MULTI():
    """Loads preprocessed dataset files from pre-defined path, and splits into inputs and output.

    Returns:
        x_train, x_test, y_train, y_test: preprocessed splitted dataset
    """
    try:
        print("( 1 ) Reading Preprocessed CSV files..")
        print(f"Train Dataset: {train_dataset} is used!")
        print(f"Test Dataset: {test_dataset} is used!")
        train_multi = pd.read_csv(train_dataset)  # selected in code started line 17 if els
        print("\t Training dataset loaded..")
        test_multi = pd.read_csv(test_dataset)  # selected in code started line 17 if els
        print("\t Testing dataset loaded..\n")

        print("( 2 ) Loading done, splitting into X and Y..")
        x_train_multi, y_train_multi = train_multi.drop(["attack_cat"], axis=1), train_multi["attack_cat"]
        x_test_multi, y_test_multi = test_multi.drop(["attack_cat"], axis=1), test_multi["attack_cat"]
        print('\t ( 2.1 ) x_train Shape: ', '\t', x_train_multi.shape)
        print('\t ( 2.2 ) y_train Shape: ', '\t', y_train_multi.shape)
        print('\t ( 2.3 ) x_test Shape: ', '\t', x_test_multi.shape)
        print('\t ( 2.4 ) y_test Shape: ', '\t', y_test_multi.shape)

        print("( 3 ) Done!")
        print("PS! Import with: x_train_multi, x_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI")

    except:
        print("Could not load dataset, try again..")
    return x_train_multi, x_test_multi, y_train_multi, y_test_multi


X_train_multi, X_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()


#max_depth 20 19 18 17 16 15 14
# Create the parameter grid based on the results of random search
print("grid search start...")
param_grid = {
    'bootstrap': [True],
    'max_depth': [20,19,18,17,16,15,14],
    'max_features':["sqrt", "log2", None],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    #'min_samples_leaf': [1],
    'min_samples_split': [2, 4, 6, 8, 10],
    #'min_samples_split': [2],
    'n_estimators': [100, 200, 300, 500, 1000],
    #'n_estimators': [100, 200],
    'criterion':["gini", "entropy", "log_loss"],
    'min_weight_fraction_leaf':[0.0],
    'max_leaf_nodes':[None],
    'min_impurity_decrease':[0.0],
    'random_state':[None],
    'verbose':[0],
    'warm_start':[False],
    'class_weight':["balanced", "balanced_subsample", None],
    'ccp_alpha':[0.0],
    'max_samples':[None],
}

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2, error_score='raise')

# Fit the grid search to the data
grid_search.fit(X_train_multi, y_train_multi)

grid_search.best_params_

best_grid = grid_search.best_estimator_

print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
#print('Improvement of {:0.2f}%.'.format( 100 * (grid_test_accuracy - base_accuracy) / base_accuracy))
print("--------------------------------------------------------------------------")
#print(f"Training accuracy: \t{grid_train_accuracy}\nTest accuracy: \t\t{grid_test_accuracy}")
