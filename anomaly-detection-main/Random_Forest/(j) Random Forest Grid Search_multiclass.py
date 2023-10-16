#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
import pandas as pd
sys.path.append("..")
from Functions.UNSW_DF import DF_XY_MULTI


X_train_multi, X_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()
"""
# to map it to the variable names in the evaluate
#test_features = X_test_multi
#test_labels = y_test_multi


# In[6]:
# funciotn to get the evaluation of the model
def evaluate_te(model, y_test_multi, X_test_multi):
    predictions = model.predict(y_test_multi)
    errors = abs(predictions - X_test_multi)
    mape = 100 * np.mean(errors / X_test_multi)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

def evaluate_tr(model, y_train_multi, X_train_multi):
    predictions = model.predict(y_train_multi)
    errors = abs(predictions - X_train_multi)
    mape_tr = 100 * np.mean(errors / X_train_multi)
    accuracy_tr = 100 - mape_tr
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy_tr))

    return accuracy_tr
"""

#max_depth 20 19 18 17 16 15 14
# Create the parameter grid based on the results of random search
print("grid search start...")
param_grid = {
    'bootstrap': [True],
    'max_depth': [9,],#14,15,16],#17,18,19],
    'max_features':["sqrt", "log2", None],
    'min_samples_leaf': [1],#, 2, 3, 4, 5],
    #'min_samples_leaf': [1],
    'min_samples_split': [2],# 4, 6, 8, 10],
    #'min_samples_split': [2],
    #'n_estimators': [100, 200, 300, 500, 1000],
    'n_estimators': [14],#10,11,12,13,14],#,15,16,17,18,19,20,21],
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


# In[7]:


# Fit the grid search to the data
grid_search.fit(X_train_multi, y_train_multi)


# In[8]:


grid_search.best_params_


# In[13]:


best_grid = grid_search.best_estimator_
#grid_train_accuracy = evaluate_tr(best_grid, X_train_multi, y_train_multi)
#grid_test_accuracy = evaluate_te(best_grid, X_test_multi, y_test_multi)

#train_accuracy = round(metrics.accuracy_score(y_train_multi, y_pred_train), 5)
#test_accuracy = round(metrics.accuracy_score(y_test_multi, y_pred_test), 5)
# In[14]:

print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
#print('Improvement of {:0.2f}%.'.format( 100 * (grid_test_accuracy - base_accuracy) / base_accuracy))
print("--------------------------------------------------------------------------")
#print(f"Training accuracy: \t{grid_train_accuracy}\nTest accuracy: \t\t{grid_test_accuracy}")

# In[ ]:




