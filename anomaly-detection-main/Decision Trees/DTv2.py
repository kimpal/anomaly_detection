#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
import sys
sys.path.append("..")
from Functions.UNSW_DF import DF_XY, DF_preprocessed_traintest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
X_train, X_test, y_train, y_test = DF_XY()

# importing Dataset
train, test = DF_preprocessed_traintest()

# In[2]:

start_time = time.time()
#test = pd.read_csv("../Dataset/test_pp3.csv", sep=',', header=0)
#train = pd.read_csv("../Dataset/train_pp3.csv", sep=',', header=0)


# In[3]:


X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]


# In[6]:


clg = DecisionTreeClassifier()
clg = clg.fit(X_train, y_train)
y_prede = clg.predict(X_test)


# In[8]:


print("Accuracy Score: \t", metrics.accuracy_score(y_test, y_prede))
print("F1 Score: \t\t", metrics.f1_score(y_test, y_prede))
print("Precision Score: \t", metrics.precision_score(y_test, y_prede))
print("Recall Score: \t\t", metrics.recall_score(y_test, y_prede))
print("------------------------------------------------")
print(metrics.classification_report(y_test, y_prede))
print("Classification report: ")
print(metrics.classification_report(y_test, y_prede))


# In[9]:


clf = DecisionTreeClassifier(criterion="gini", min_samples_leaf=20, min_samples_split=14, max_features=16, ccp_alpha=0.00000000000000002)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[10]:


print("Accuracy Score: \t", metrics.accuracy_score(y_test, y_pred))
print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred))
print("Precision Score: \t", metrics.precision_score(y_test, y_pred))
print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred))
print("------------------------------------------------")
print(metrics.classification_report(y_test, y_pred))
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred))


# In[34]:
"""
print("Gridsearch------------------")

params = {"criterion":["gini","entropy"],
          "min_samples_split": range(1,15), 
          "min_samples_leaf": range(1,20),
          "splitter":["best", "random"],
          }

clf = DecisionTreeClassifier()
gs = GridSearchCV(estimator=clf, param_grid=params, 
                n_jobs=-1, cv=5,)
gs.fit(X_train, y_train)

gs.best_estimator_.fit(X_train, y_train)
y_pred = gs.best_estimator_.predict(X_test)
y_true = y_test


# In[35]:


print(gs.best_params_)
print(gs.best_estimator_)
print(gs.best_score_)
print("Accuracy test", metrics.accuracy_score(y_test, y_pred))
print("grid search fish")
"""

# In[11]:


clf_ = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.00002)
clf_.fit(X_train,y_train)
y_train_pred = clf_.predict(X_train)
y_test_pred = clf_.predict(X_test)

print("Accuracy Score: \t", metrics.accuracy_score(y_test, y_pred))
print("F1 Score: \t\t", metrics.f1_score(y_test, y_pred))
print("Precision Score: \t", metrics.precision_score(y_test, y_pred))
print("Recall Score: \t\t", metrics.recall_score(y_test, y_pred))
print("------------------------------------------------")
print(metrics.classification_report(y_test, y_pred))
print("Classification report: ")
print(metrics.classification_report(y_test, y_pred))

#timer
elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")
