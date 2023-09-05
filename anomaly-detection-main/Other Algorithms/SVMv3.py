#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

import sys
sys.path.append("..")
from Functions.UNSW_DF import *


# In[5]:


train, test = DF_preprocessed_traintest()
X_train, X_test, y_train, y_test = DF_XY()


# In[ ]:


train.drop(train.index[10000:175341], axis=0, inplace=True)
test.drop(test.index[5000:82332], axis=0, inplace=True)
print(f"Train shape:\t {train.shape}\nTest shape:\t {test.shape}")


# In[ ]:


error = []
C_value = []
accuracy_score = []
precision_score = []
F1_score = []
recall_score = []

def SVM_predict(c_start, c_end, svm_kernel, svm_degree):
    """Predicts an SVM model with given arguments

    Args:
        c_start (int): C value start
        c_end (int): C value end
        svm_kernel (string): SVM kernel given in a string format: i.e. linear, poly, rbf
        svm_degree (int): default=3, Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    """
    c_start = c_start
    c_end += 1
    
    for c in range(c_start, c_end):
        SVM_model = SVC(kernel=svm_kernel, C = c, degree=svm_degree)
        SVM_model.fit(X_train, y_train)
        pred_i = SVM_model.predict(X_test)
        
        # Appending values to list
        error.append(np.mean(pred_i != y_test))
        C_value.append(c)
        accuracy_score.append(metrics.accuracy_score(y_test, pred_i))
        precision_score.append(metrics.precision_score(y_test, pred_i))
        F1_score.append(metrics.f1_score(y_test, pred_i))
        recall_score.append(metrics.recall_score(y_test, pred_i))
    


# In[ ]:


# Calling the function created above.
SVM_predict(c_start=1, c_end=15, svm_kernel="rbf", svm_degree=3)


# In[ ]:


# Creating a dataframe and saving to file
# dictionary of lists 
dict = {
        'C': C_value, 
        'accuracy': accuracy_score, 
        'precision': precision_score,
        'F1': F1_score,
        'recall': recall_score,
        'error': error
        }

df = pd.DataFrame(dict, index=C_value)
df.set_index("C", inplace = True)
df


# In[ ]:


# EXPORT AS CSV when done.
df.to_csv('SVM_scores(1-15)_kernel_z.csv')

