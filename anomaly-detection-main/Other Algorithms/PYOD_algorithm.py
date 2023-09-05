#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from Functions.UNSW_DF import *
from pyod.models.abod import ABOD
from pyod.utils.data import generate_data

x_train, x_test, y_train, y_test = DF_XY()

# importing Dataset
train, test = DF_preprocessed_traintest()
# In[4]:


X_train, y_train, X_test, y_test = \
        generate_data(n_train=200,
                      n_test=100,
                      n_features=5,
                      contamination=0.1,
                      random_state=3) 
X_train = X_train * np.random.uniform(0, 1, size=X_train.shape)
X_test = X_test * np.random.uniform(0,1, size=X_test.shape)


# In[5]:


clf_name = 'ABOD'
clf = ABOD()
clf.fit(X_train)
test_scores = clf.decision_function(X_test)

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
print(f'{clf_name} ROC:{roc}, precision @ rank n:{prn}')


# In[7]:


from pyod.models.copod import COPOD
clf_name = 'COPOD'
clf = COPOD()
clf.fit(X_train)
test_scores = clf.decision_function(X_test)

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
prn = round(precision_n_scores(y_test, test_scores), ndigits=4)
print(f'{clf_name} ROC:{roc}, precision @ rank n:{prn}')

