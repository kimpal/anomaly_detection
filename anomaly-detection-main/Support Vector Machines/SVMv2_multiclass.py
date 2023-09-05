#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("..")

# In[2]:


#importing the datasets
test = pd.read_csv("../Dataset/test_pp3_multi.csv", sep=',', header=0)
train = pd.read_csv("../Dataset/train_pp3_multi.csv", sep=',', header=0)

print(test.shape)
print(train.shape)


#splitting the datasets into input and output
X_train = train.drop(['attack_cat'], axis=1)
X_test = test.drop(['attack_cat'], axis=1)
y_train = train.loc[:, ['attack_cat']]
y_test = test.loc[:, ['attack_cat']]


print(np.unique(y_train))


# In[11]:

"""
model = SVC(kernel = "linear", C=100, gamma = "scale")
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
print("Accuracy Score: \t", metrics.accuracy_score(y_test, y_pred))
print("Accuracy Score: \t", metrics.accuracy_score(y_train, y_pred_train))
print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred))
print("Precision Score: \t", metrics.precision_score(y_test, y_pred))
print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred))
print("------------------------------------------------")
print(metrics.classification_report(y_test, y_pred))
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred))
"""

# In[7]:

"""
model = SVC(kernel='rbf', C=250, gamma="auto")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
print("Accuracy Score: \t", metrics.accuracy_score(y_test, y_pred))
print("Accuracy Score: \t", metrics.accuracy_score(y_train, y_pred_train))
print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred))
print("Precision Score: \t", metrics.precision_score(y_test, y_pred))
print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred))
print("------------------------------------------------")
print(metrics.classification_report(y_test, y_pred))
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred))


# In[14]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
"""

# In[5]:
# C 500
print("grid search start..")
model = SVC()
param_grid = {"C": [1.0],
             "gamma": ['scale', 'auto',1, 0.1, 0.01, 0.001, 0.0001],
             "kernel": ["rbf", "linear", "poly", "", "sigmoid", "precomputed"],
              "degree":[3],
              "coef0":[0.0],
              "shrinking":[True],
              "probability":[False],
              "tol":[1e-3],
              "cache_size":[200],
              "class_weight":['balanced', None],
              "verbose":[False],
              "max_iter":[-1],
              "decision_function_shape":["ovo", "ovr"],
              "break_ties":[False],
              "random_state":[None]
             }

grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(X_train, y_train.values.ravel())

grid.best_params_

# In[6]:
#y_pred = grid.predict(X_test)
best_grid = grid.best_estimator_
grid_train_accuracy = model.evaluate(best_grid, X_train, y_train)
grid_test_accuracy = model.evaluate(best_grid, X_test, y_test)
weighted_f1 = round(metrics.f1_score(y_test, grid_test_accuracy, average='weighted'), 5)
print(f"Training accuracy: \t{grid_train_accuracy}\nTest accuracy: \t\t{grid_test_accuracy}\nWeighted F1-score: \t{weighted_f1}")

print("--------------------------------------------------------------------------")

print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))
