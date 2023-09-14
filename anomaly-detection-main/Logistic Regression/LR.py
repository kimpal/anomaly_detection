#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


import sys
sys.path.append("..")
from Functions.UNSW_DF_Multi_label import DF_XY_Multi


# In[2]:


X_train, X_test, y_train, y_test = DF_XY_Multi()

# In[36]:

# 300, 500, l2, lbfgs

logreg = LogisticRegression(max_iter = 100, C = 100, penalty = "l2", solver = "sag", multi_class='multinomial')
logreg.fit(X_train, y_train.ravel())
y_pred_test = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)
print("Accuracy Score test: \t", metrics.accuracy_score(y_test, y_pred_test))
print("Accuracy Score train: \t", metrics.accuracy_score(y_train, y_pred_train))
print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred_test, average='macro', zero_division= 0.0))
print("Precision Score: \t", metrics.precision_score(y_test, y_pred_test, average='macro', zero_division=0.0))
print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred_test, average='macro', zero_division= 0.0))
print("------------------------------------------------")
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred_test, zero_division= 0.0))

print("Accuracy test", metrics.accuracy_score(y_test, y_pred_test))
print("......................................")
print("Accuracy train", metrics.accuracy_score(y_train, y_pred_train))
