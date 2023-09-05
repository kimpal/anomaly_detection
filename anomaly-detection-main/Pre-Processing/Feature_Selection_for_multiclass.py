#!/usr/bin/env python
# coding: utf-8

# # Feature Selection

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("..")
from Functions.UNSW_DF import *

train, test = DF_original_traintest()


# In[2]:


train.drop(["attack_cat"], axis=1, inplace = True)
test.drop(["attack_cat"], axis=1, inplace = True)


# In[3]:


# Defining the columns that need to be label encoded.
cols = ['proto', 'service', 'state']
le = preprocessing.LabelEncoder()

# Label encoding the columns for the test and training set
test[cols] = test[cols].apply(le.fit_transform)
train[cols] = train[cols].apply(le.fit_transform)


# In[4]:


X_train, y_train = train.drop(["label"], axis=1), train["label"]
X_test, y_test = test.drop(["label"], axis=1), test["label"]


# In[5]:


# OPTIONAL 1: Applying Min Max Scaler on X
mm_scaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_scaler.fit_transform(X_train)
X_test_minmax = mm_scaler.fit_transform(X_test)


# In[6]:


# OPTIONAL 2: Applying StandardScaler on X
ss = StandardScaler()
X_train_ss = pd.DataFrame(ss.fit_transform(X_train),columns = X_train.columns)


# ## Recursive Feature Elimination (RFE)

# In[7]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[8]:


# Selecting the Best important features according to Logistic Regression
rfe_selector = RFE(estimator=LogisticRegression(solver='lbfgs', max_iter=500), step = 1, verbose = 1,)


# In[9]:


rfe_fit = rfe_selector.fit(X_train_ss, y_train)


# In[10]:


X_train_ss.columns[rfe_selector.get_support()]


# In[11]:


print("Num Features: %d" % rfe_fit.n_features_)
print("Selected Features: %s" % rfe_fit.support_)
print("Feature Ranking: %s" % rfe_fit.ranking_,)


# ## Univariate Feature Selection with SelectKBest

# In[12]:


from sklearn.feature_selection import SelectKBest, mutual_info_regression


# In[13]:


# Select top "all" features based on mutual info regression
selector = SelectKBest(mutual_info_regression, k = "all")
selector.fit(X_train_ss, y_train)
X_train_ss.columns[selector.get_support()]


# ## Sequential Feature Selection (SFS)

# In[14]:


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression


# In[15]:


#Selecting the Best important features according to Logistic Regression
sfs_selector = SequentialFeatureSelector(estimator=LogisticRegression(max_iter=100, verbose=1, n_jobs=-1), cv =10, direction ='backward')


# In[ ]:


sfs_selector.fit(X_train_ss, y_train)


# In[ ]:


X_train_ss.columns[sfs_selector.get_support()]


# ## Testing

# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[ ]:


# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X_train, y_train)

