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

# In[ ]:

print("Grid search starting...")

lr = LogisticRegression()
solvers = ['sag']#, 'sag','saga']
penalty = ['l2']#['None', 'l1', 'l2', 'elasticnet']
c_values = [100, 10, 1, 0.1]
max_iter = [100, 200 , 1000]

grid = dict(solver=solvers, penalty = penalty ,C=c_values, max_iter=max_iter)
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=lr, param_grid=grid, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X_train, y_train)

print("best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



# In[ ]:

y_pred_test = grid_search.predict(X_test)
y_pred_train = grid_search.predict(X_train)

print("Accuracy test", metrics.accuracy_score(y_test, y_pred_test))
print("......................................")
print("Accuracy train", metrics.accuracy_score(y_train, y_pred_train))

# In[]:

df = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    
# EXPORT AS CSV when done.
df.to_csv('LR_Multi_gridsearch_10_classes_solvers.csv') 