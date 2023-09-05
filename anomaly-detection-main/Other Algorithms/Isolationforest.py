#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import sys
sys.path.append("..")
from Functions.UNSW_DF import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


# In[2]:
#X_train, X_test, y_train, y_test = DF_XY()

# importing Dataset
#train, test = DF_preprocessed_traintest()


test = pd.read_csv("../../Anomaly-Detection-main/Dataset/UNSW_NB15_testing-set.csv", sep=',', header=0)
train = pd.read_csv("../../Anomaly-Detection-main/Dataset/UNSW_NB15_training-set.csv", sep=',', header=0)

combined_trainTest = pd.concat([train, test]).drop(['id'], axis=1)


# In[3]:


combined_trainTest.head()


# In[4]:


cols = ['proto', 'service', 'state']
le = preprocessing.LabelEncoder()

combined_trainTest[cols] = combined_trainTest[cols].apply(le.fit_transform)
combined_trainTest.head()


# In[5]:


X = combined_trainTest.drop(['label', 'attack_cat'], axis=1)
y = combined_trainTest.loc[:, ['label']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)
X_train.head(9)


# In[6]:


n = 40
#rfe = RFE(DecisionTreeClassifier(),n).fit(X_train, y_train)
rfe = RFE(estimator=DecisionTreeClassifier(),n_features_to_select=5)
rfe.fit(X,y)

di = np.where(rfe.support_==True)[0]
list = X_train.columns.values[di]
X_train_RFE, X_test_RFE = X_train[list], X_test[list]
print('new shape', X_train_RFE.shape)


# In[7]:


params = {'max_depth': [2,4,6,8,10], 
          'min_samples_split': [2,3,4,5], 
          'min_samples_leaf': [1,2,4,6,8,10]}

clf = DecisionTreeClassifier()
gs = GridSearchCV(estimator=clf, param_grid=params, scoring="accuracy",
                cv=10, return_train_score=True, verbose=1)
gs.fit(X_train_RFE, y_train)

gs.best_estimator_.fit(X_train_RFE, y_train)
y_pred = gs.best_estimator_.predict(X_test_RFE)
y_true = y_test



# In[8]:


print("Test accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[9]:


print(classification_report(y_test, y_pred))


# In[ ]:




