#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier

# In[18]:

import time
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from Functions.UNSW_DF import *

# importing random forest classifier from assemble module
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
X_train, X_test, y_train, y_test = DF_XY()

# importing Dataset
train, test = DF_preprocessed_traintest()


# In[2]:


dataframe = pd.merge(train, test)
X = dataframe.drop(["label"], axis=1)
y= dataframe["label"]


# ## Creating the classifier

# In[ ]:
start_time = time.time()
params = {"criterion": "entropy",
              "bootstrap":True, 
              "n_estimators": 200,
              "max_depth": 50,
              "min_samples_split":2,
              "min_samples_leaf": 1,
              "n_jobs": -1}


# In[ ]:


# define the model
model = RandomForestClassifier(**params)
model.set_params(**params)

# fit the model on the whole dataset
model.fit(X_train, y_train)

# performing predictions on the traing and test dataset
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# In[ ]:


train_accuracy = round(metrics.accuracy_score(y_train, y_pred_train), 5)
test_accuracy = round(metrics.accuracy_score(y_test, y_pred_test), 5)
f1 = round(metrics.f1_score(y_test, y_pred_test), 5)
precision = round(metrics.precision_score(y_test, y_pred_test), 5)
recall = round(metrics.recall_score(y_test, y_pred_test), 5)

print(f"Training accuracy: \t{train_accuracy}\nTest accuracy: \t\t{test_accuracy}\nF1-score: \t\t{f1}\nprecision-score: \t{precision}\nrecall-score: \t\t{recall}\n")

elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime {elapsed_time}s")

"""""
# ## Classifier Experiment 1

# In[3]:


# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
import time


# In[4]:


params = {"criterion": "entropy",
              "bootstrap":True,
              "n_estimators": 200,
              "max_depth": 50,
              "min_samples_split":2,
              "min_samples_leaf": 1,
              "n_jobs": -1}


# In[5]:


# define the model
model = RandomForestClassifier(n_jobs=-1)

# define the model
model = RandomForestClassifier(**params)
model.set_params(**params)

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10,
                             n_repeats=3,
                             random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')


# In[6]:


n_scores


# In[7]:


# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
"""""

# In[ ]:





# # Classifier Experiment 2

# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


# In[ ]:


# define the model
#model = RandomForestClassifier(n_jobs=-1)


# In[ ]:


#scores = cross_validate(model, X_train, y_train, cv=30,
#                        scoring=('accuracy', 'precision', 'recall', 'f1'),
#                        return_train_score=True,
#                        verbose=1)


# In[ ]:


#scores


# In[ ]:


#df = pd.DataFrame.from_dict(scores, orient='columns')
#df


# # CROSS VAL 3
