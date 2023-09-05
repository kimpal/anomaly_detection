#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import time
sys.path.append("..")
from Functions.UNSW_DF import *

import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report

x_train, x_test, y_train, y_test = DF_XY() #XY_import()


# In[2]:


train, test = DF_preprocessed_traintest()


# In[ ]:


#test = pd.read_csv("../../Anomaly-Detection-main/Dataset/test_pp3.csv", sep=',', header=0)
#train = pd.read_csv("../../Anomaly-Detection-main/Dataset/train_pp3.csv", sep=',', header=0)


# In[ ]:


X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]

y_train.shape
#y_train = y_train.ravel()

# In[ ]:


logreg = LogisticRegression(max_iter = 10000)
logreg.fit(X_train, y_train.values.ravel())
y_pred = logreg.predict(X_test)


# In[ ]:


print("Test accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[ ]:


lr = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['none', 'l1', 'l2', 'elasticnet']
c_values = [10, 1.0, 0.1, 0.01]
max_iter = [100,100,1000]

grid = dict(solver=solvers, penalty=penalty, C=c_values, max_iter=max_iter)
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=lr, n_jobs=-1, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X_train, y_train.values.ravel())
y_pred = grid_search.predict(X_test)


# In[ ]:


print("best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Accuracy test", metrics.accuracy_score(y_test, y_pred))
print("......................................")
print("Accuracy train", metrics.accuracy_score(y_train, y_pred))

