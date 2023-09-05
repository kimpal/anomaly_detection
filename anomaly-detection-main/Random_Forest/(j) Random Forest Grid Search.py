#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sys 
sys.path.append("..")
from Functions.UNSW_DF import *


X_train, X_test, y_train, y_test = DF_XY()


# In[6]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [40, 50, 60, 70, 80],
    'max_features': ["auto", "sqrt", "log2"],
    'min_samples_leaf': [2, 3, 4, 5],
    'min_samples_split': [4, 6, 8, 10],
    'n_estimators': [200, 300, 500, 1000]
}

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)


# In[7]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train)


# In[8]:


grid_search.best_params_


# In[13]:


best_grid = grid_search.best_estimator_
grid_accuracy = grid_search.evaluate(best_grid, X_test, y_test)


# In[14]:


print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# In[ ]:




