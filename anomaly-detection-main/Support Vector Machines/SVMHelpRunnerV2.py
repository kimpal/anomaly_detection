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
test = pd.read_csv("test_preprocessed.csv", sep=',', header=0)
train = pd.read_csv("train_preprocessed.csv", sep=',', header=0)

test.shape


# In[3]:


#splitting the datasets into input and output
X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]


# In[5]:


model = SVC(kernel = "sigmoid", C=100, gamma = "scale")
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

