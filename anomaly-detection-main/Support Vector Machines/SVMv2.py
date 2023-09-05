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


# In[2]:


#importing the datasets
test = pd.read_csv("test_pp3.csv", sep=',', header=0)
train = pd.read_csv("train_pp3.csv", sep=',', header=0)

test.shape



# In[3]:


#splitting the datasets into input and output
X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]



# In[4]:


np.unique(y_train)


# In[11]:


model = SVC(kernel = "linear", C=100, gamma = "scale")
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


# In[7]:


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


# In[5]:


model = SVC()
param_grid = {"C": [500],
             "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
             "kernel": ["rbf"],
             }

grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(X_train, y_train)


# In[6]:


print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))

