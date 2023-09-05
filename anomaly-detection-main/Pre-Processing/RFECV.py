#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
np.random.seed(5)


# In[9]:


test = pd.read_csv("../Dataset/UNSW_NB15_testing-set.csv", sep=',', header=0)
train = pd.read_csv("../Dataset/UNSW_NB15_training-set.csv", sep=',', header=0)


# In[10]:


X_train = train.drop(['attack_cat','label', 'id'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]


# In[11]:


cols = ['proto', 'service', 'state']
le = preprocessing.LabelEncoder()


# In[12]:


X_test[cols] = X_test[cols].apply(le.fit_transform)
X_train[cols] = X_train[cols].apply(le.fit_transform)


# In[13]:


rf = RandomForestClassifier()
#X, y = make_classification(n_samples=10000, n_features=44, )
cv = StratifiedKFold(10)
min_features_to_select = 1
visualizer = RFECV(estimator = rf, step = 1, cv=cv, scoring='accuracy', min_features_to_select=min_features_to_select)

visualizer.fit(X_train, y_train)
print("optimal number of features : %d" % visualizer.n_features_)
print("feauture", visualizer.n_features_)
print("best features", visualizer.ranking_)
print("Support", visualizer.support_ )
print("n_features_in", visualizer.n_features_in_)
#print(visualizer.cv_results_)
print("please print the right feature", visualizer.grid_scores_)


# In[ ]:


plt.figure()
plt.xlabel("number of features selected")
plt.ylabel("cross validation score(accuracy)")
plt.plot(range(min_features_to_select, len(visualizer.grid_scores_) + min_features_to_select), visualizer.grid_scores_)
plt.show


# In[34]:


df_features = pd.DataFrame(columns = ['feature', 'support', 'ranking'])

for i in range(X_train.shape[1]):
    row = {'feature': X_train.columns[i], 
           'support': visualizer.support_[i], 
           'ranking': visualizer.ranking_[i],
           #'cv_results': visualizer.cv_results_[i]
           'Grid_score': visualizer.grid_scores_[i]
           }
    df_features = df_features.append(row, ignore_index=True)
    
df_features = df_features.sort_values(by='ranking')


# In[35]:


df_features.to_csv("RFECV_results.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


rfecv = RFECV(estimator=dtree, step=1, cv=4, verbose=1,
              scoring='accuracy', n_jobs= -1)
rfecv.fit(X, y)


# In[ ]:


rfecv.transform(X)


# In[ ]:


print(); print(rfecv)
print();print("Optimal number of features: {}".format(rfecv.n_features_))
print();print(np.where(rfecv.support_ == False)[0])
      


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


dataset = pd.read_csv("RFECV_results.csv")


# In[4]:


dataset= dataset.round(5)
dataset.to_csv("RFECV_results_rounded.csv", index = False)


# In[ ]:




