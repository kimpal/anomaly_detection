#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  


# In[2]:


#importing the datasets
test = pd.read_csv("test_preprocessed.csv", sep=',', header=0)
train = pd.read_csv("train_preprocessed.csv", sep=',', header=0)


# In[3]:


#splitting the datasets into input and output
X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]


# In[4]:


#apply different kernel to transform the data
#Kernels = ['Polynomial', 'RBF', 'Sigmoid', 'Linear']
#def getClassifier(kerneltype):
#    if kerneltype == 0:
#        return SVC(kernel='poly', degree=8, gamma="auto")
#    elif kerneltype == 1:
#        return SVC(kernel='rbf', gamma="auto")
#    elif kerneltype == 2:
#        return SVC(kernel='sigmoid', gamma="auto")
#    elif kerneltype == 3:
#        return SVC(kernel='linear', gamma="auto")


# In[5]:


#for i in range(4):
#    print("step1")
#    svclassifier = getClassifier(i)
#    print("step2")
#    svclassifier.fit(X_train, y_train.values.ravel())
#    print("step3")
#    y_pred = svclassifier.predict(X_test)
#    print("Evaluation:", Kernels[i], "kernel")
#    print(classification_report(y_test, y_pred))


# In[6]:


model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:




