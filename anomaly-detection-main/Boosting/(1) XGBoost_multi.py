#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[19]:


train = pd.read_csv("../Dataset/train_pp3_multi.csv")
test = pd.read_csv("../Dataset/test_pp3_multi.csv")

x_train, y_train = train.drop(["attack_cat"], axis=1), train["attack_cat"]
x_test, y_test = test.drop(["attack_cat"], axis=1), test["attack_cat"]
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
y_pred_train = model.predict(x_train)
predictions_train = [round(value) for value in y_pred_train]

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]


# In[24]:


# evaluate predictions
accuracy_train = accuracy_score(y_train, predictions_train)
print("Accuracy: %.2f%%" % (accuracy_train * 100.0))
accuracy_test = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy_test * 100.0))

