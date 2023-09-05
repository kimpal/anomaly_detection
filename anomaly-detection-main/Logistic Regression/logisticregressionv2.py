#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time
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
from Functions.UNSW_DF import *


# In[ ]:
start_time = time.time()

train, test = DF_preprocessed_traintest()
X_train, X_test, y_train, y_test = DF_XY()


# In[ ]:


print(y_test.shape)
print(y_test.shape)

y_train.ravel()

# In[ ]:

"""
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())
y_pred_test = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)


# In[ ]:


#print("Accuracy Score test: \t", metrics.accuracy_score(y_test, y_pred_test))
#print("Accuracy Score train: \t", metrics.accuracy_score(y_train, y_pred_train))
#print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred_test))
#print("Precision Score: \t", metrics.precision_score(y_test, y_pred_test))
#print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred_test))
#print("------------------------------------------------")
#print("Classification report: ")
#print(metrics.classification_report(y_test, y_pred_test))


# In[ ]:


#logreg = LogisticRegression(max_iter = 1250)
#logreg.fit(X_train, y_train.values.ravel())
#y_pred_test = logreg.predict(X_test)
#y_pred_train = logreg.predict(X_train)


# In[ ]:


#print("Accuracy Score test: \t", metrics.accuracy_score(y_test, y_pred_test))
#print("Accuracy Score train: \t", metrics.accuracy_score(y_train, y_pred_train))
#print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred_test))
#print("Precision Score: \t", metrics.precision_score(y_test, y_pred_test))
#print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred_test))
#print("------------------------------------------------")
#print("Classification report: ")
#print(metrics.classification_report(y_test, y_pred_test))

"""
# In[ ]:

#300, 500, l2, lbfgs

logreg = LogisticRegression(max_iter = 1000, C = 100, penalty = "l1", solver = "liblinear")
logreg.fit(X_train, y_train.ravel())
y_pred_test = logreg.predict(X_test)
y_pred_train = logreg.predict(X_train)
print("Accuracy Score test: \t", metrics.accuracy_score(y_test, y_pred_test))
print("Accuracy Score train: \t", metrics.accuracy_score(y_train, y_pred_train))
print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred_test))
print("Precision Score: \t", metrics.precision_score(y_test, y_pred_test))
print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred_test))
print("------------------------------------------------")
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred_test))


# In[ ]:

"""
#print("Accuracy Score test: \t", metrics.accuracy_score(y_test, y_pred_test))
#print("Accuracy Score train: \t", metrics.accuracy_score(y_train, y_pred_train))
#print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred_test))
#print("Precision Score: \t", metrics.precision_score(y_test, y_pred_test))
#print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred_test))
#print("------------------------------------------------")
#print("Classification report: ")
#print(metrics.classification_report(y_test, y_pred_test))


# In[ ]:

print("logisticRegression start...")
lr = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['none', 'l1', 'l2', 'elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
max_iter = [10,100,1000]

grid = dict(solver=solvers, penalty=penalty, C=c_values, max_iter=max_iter)
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=lr, n_jobs=-1, param_grid=grid, cv=cv, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
y_pred_train = grid_search.predict(X_train)


# In[ ]:


print("best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Accuracy test", metrics.accuracy_score(y_test, y_pred))
print("......................................")
print("Accuracy train", metrics.accuracy_score(y_train, y_pred_train))
"""
# In[ ]:
"""
print("Grid search starting..")

lr = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['none', 'l1', 'l2', 'elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
max_iter = [1000]

grid = dict(solver=solvers, penalty=penalty, C=c_values, max_iter=max_iter)
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=lr, param_grid=grid, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
y_pred_train = grid_search.predict(X_train)

print("best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Accuracy test", metrics.accuracy_score(y_test, y_pred))
print("......................................")
print("Accuracy train", metrics.accuracy_score(y_train, y_pred_train))
"""
elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime {elapsed_time}s")