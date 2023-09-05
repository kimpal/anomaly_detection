#!/usr/bin/env python
# coding: utf-8

# # Adapitive Boosting (Adaboost)

# In[13]:

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import sys
sys.path.append("..")
from Functions.UNSW_DF import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# In[14]:


x_train, x_test, y_train, y_test = DF_XY()


# In[15]:


# define the model
model = AdaBoostClassifier()


# In[16]:


# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')


# In[23]:


# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# In[ ]:





# In[ ]:





# In[ ]:




