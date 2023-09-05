#!/usr/bin/env python
# coding: utf-8

# # (1) Naive Bayes Classifier

# In[6]:


import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from Functions.UNSW_DF import *
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = DF_XY()

# importing Dataset
train, test = DF_preprocessed_traintest()

# In[7]:
# convert to multiclass

#train = pd.read_csv("Dataset/train_pp3.csv")
#test = pd.read_csv("Dataset/test_pp3.csv")

x_train, y_train = train.drop(["label"], axis=1), train["label"]
x_test, y_test = test.drop(["label"], axis=1), test["label"]
print('X_train Shape: ', '\t', x_train.shape)
print('y_train Shape: ', '\t', y_train.shape)
print('X_test Shape: ', '\t\t', x_test.shape)
print('y_test Shape: ', '\t\t', y_test.shape)


# In[8]:


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


# In[9]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[10]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy of the model:",metrics.accuracy_score(y_test, y_pred)*100, '%')


# In[ ]:




