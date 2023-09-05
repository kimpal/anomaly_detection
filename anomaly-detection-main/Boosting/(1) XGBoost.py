#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[19]:


train = pd.read_csv("../Dataset/train_pp3.csv")
test = pd.read_csv("../Dataset/test_pp3.csv")

x_train, y_train = train.drop(["label"], axis=1), train["label"]
x_test, y_test = test.drop(["label"], axis=1), test["label"]
print('X_train Shape: ', '\t', x_train.shape)
print('y_train Shape: ', '\t', y_train.shape)
print('X_test Shape: ', '\t\t', x_test.shape)
print('y_test Shape: ', '\t\t', y_test.shape)


# In[20]:


model = XGBClassifier()


# In[21]:


model.fit(x_train, y_train)


# In[22]:


print(model)


# In[23]:


y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]


# In[24]:


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

