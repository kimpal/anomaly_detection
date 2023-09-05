#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# In[8]:


test = pd.read_csv("../Dataset/test_preprocessed.csv", sep=',', header=0)
train = pd.read_csv("../Dataset//train_preprocessed.csv", sep=',', header=0)
df = pd.DataFrame(train)
train.shape


# In[ ]:


X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]


# In[ ]:


params = {'max_depth': [2,4,6,8,10], 
          'min_samples_split': [2,3,4,5], 
          'min_samples_leaf': [1,2,4,6,8,10]}

clf = DecisionTreeClassifier()
gs = GridSearchCV(estimator=clf, param_grid=params, scoring="accuracy",
                n_jobs=-1, cv=10, return_train_score=True, verbose=1)
gs.fit(X_train, y_train)

gs.best_estimator_.fit(X_train, y_train)
y_pred = gs.best_estimator_.predict(X_test)
y_true = y_test


# In[ ]:


#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Test accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[ ]:


model = gs.best_estimator_
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred, y_train)}')
print(f'Test score {accuracy_score(y_test_pred, y_test)}')


# In[ ]:


path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)


# In[ ]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)


# In[ ]:


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
plt.legend()
plt.show()


# In[ ]:


train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()


# In[ ]:


clf_ = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.0002)
clf_.fit(X_train,y_train)
y_train_pred = clf_.predict(X_train)
y_test_pred = clf_.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test score {accuracy_score(y_test_pred,y_test)}')


# In[ ]:





# In[ ]:




