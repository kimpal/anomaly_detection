#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbor

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Dataset loading
import sys
sys.path.append("..")
from Functions.UNSW_DF import *
x_train, x_test, y_train, y_test = DF_XY()


# In[2]:


error_test = []
K_value = []
train_acc = []
test_acc = []
precision_score = []
F1_score = []
recall_score = []

# Calculating error for K values between 1 and 200 and appending scores to lists
for i in range(1, 71):
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(x_train, y_train)
    pred_i_test = knn.predict(x_test)
    pred_i_train = knn.predict(x_train)
    
    # Appending values to list
    error_test.append(np.mean(pred_i_test != y_test))
    K_value.append(i)
    train_acc.append(metrics.accuracy_score(y_train, pred_i_train))
    test_acc.append(metrics.accuracy_score(y_test, pred_i_test))
    precision_score.append(metrics.precision_score(y_test, pred_i_test))
    F1_score.append(metrics.f1_score(y_test, pred_i_test))
    recall_score.append(metrics.recall_score(y_test, pred_i_test))
    


# In[ ]:


# Creating a dataframe and saving to file
# dictionary of lists 
dict = {
        'K': K_value,
        'train_acc': train_acc, 
        'test_acc': test_acc, 
        'precision': precision_score,
        'F1': F1_score,
        'recall': recall_score,
        'error': error_test
        }

df = pd.DataFrame(dict, index=K_value)
df.set_index("K", inplace = True)
df
# EXPORT AS CSV when done.
df.to_csv('K_SCORES(1-70).csv')


# In[ ]:


dataframe = pd.read_csv("K_SCORES(1-70).csv")


# In[ ]:


dataframe


# In[ ]:


plt.figure(figsize=(20, 12))
plt.plot(range(1, 71), error_test, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.savefig('KNN(1-70)_v3.png', dpi=40, transparent=True)
plt.show()


# In[ ]:


new_dataframe = pd.read_csv("K_SCORES(1-70).csv")
new_dataframe


# In[ ]:
#new_dataframe.set_index(['K'], inplace=False)
#new_dataframe.set_index("K", inplace=False)
new_dataframe.set_index("K", inplace=True)
new_dataframe

