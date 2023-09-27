#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
import sys
sys.path.append("..")
from Functions.UNSW_DF import DF_XY_MULTI, DF_preprocessed_traintest_multi
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
X_train_multi, X_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()

# importing Dataset
train_multi, test_multi = DF_preprocessed_traintest_multi()
# In[2]:

start_time = time.time()

X_train_multi = train_multi.drop(['attack_cat'], axis=1)
X_test_multi = test_multi.drop(['attack_cat'], axis=1)
y_train_multi = train_multi.loc[:, ['attack_cat']]
y_test_multi = test_multi.loc[:, ['attack_cat']]

# In[6]:

"""
# first model
print("first model starts...")
clg = DecisionTreeClassifier()
clg = clg.fit(X_train_multi, y_train_multi)
y_pred_train_multi = clg.predict(X_train_multi) # train prediction
y_pred_test_multi = clg.predict(X_test_multi) # test prediction

# train result
accuracy_train = round(metrics.accuracy_score(y_train_multi, y_pred_train_multi), 5)
f1_train = round(metrics.f1_score(y_train_multi, y_pred_train_multi, average='weighted'), 5)
precision_train = round(metrics.precision_score(y_train_multi, y_pred_train_multi, average='weighted'), 5)
recall_train = round(metrics.recall_score(y_train_multi, y_pred_train_multi, average='weighted'), 5)

# test result
accuracy = round(metrics.accuracy_score(y_test_multi, y_pred_test_multi), 5)
f1 = round(metrics.f1_score(y_test_multi, y_pred_test_multi, average='weighted'), 5)
precision = round(metrics.precision_score(y_test_multi, y_pred_test_multi, average='weighted'), 5)
recall = round(metrics.recall_score(y_test_multi, y_pred_test_multi, average='weighted'), 5)

# printing test accuracy
print("Accuracy train Score: \t", accuracy_train)
print("Accuracy test Score: \t", accuracy)
print("F1 train Score: \t\t", f1_train)
print("F1 test Score: \t\t", f1)
print("Precision test Score: \t", precision_train)
print("Precision test Score: \t", precision)
print("Recall test Score: \t\t", recall)
print("------------------------------------------------")
print("Classification test report: ")
print(metrics.classification_report(y_test_multi, y_pred_test_multi))
"""



# In[9]:


print("------------------------------------")
print("next model starts")
# second model
clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, min_samples_split=13, max_features=45, ccp_alpha=0.0) #splitter="best")  #ccp_alpha=0.0000002) # test whit ccp_alpha=0.0
clf = clf.fit(X_train_multi, y_train_multi)
y_pred_train_multi = clf.predict(X_train_multi)
y_pred_test_multi = clf.predict(X_test_multi)

# train scores
train_accuracy = round(metrics.accuracy_score(y_train_multi, y_pred_train_multi), 5)
train_f1 = round(metrics.f1_score(y_train_multi, y_pred_train_multi, average='weighted'), 5)
train_precision = round(metrics.precision_score(y_train_multi, y_pred_train_multi, average='weighted'), 5)
train_recall = round(metrics.recall_score(y_train_multi, y_pred_train_multi, average='weighted'), 5)

# test scores
accuracy = round(metrics.accuracy_score(y_test_multi, y_pred_test_multi), 5)
f1 = round(metrics.f1_score(y_test_multi, y_pred_test_multi, average='weighted'), 5)
precision = round(metrics.precision_score(y_test_multi, y_pred_test_multi, average='weighted'), 5)
recall = round(metrics.recall_score(y_test_multi, y_pred_test_multi, average='weighted'), 5)

print("Accuracy train Score: \t", train_accuracy)
print("Accuracy test Score: \t", accuracy)
print("F1 train Score: \t\t", train_f1)
print("F1 test Score: \t\t", f1)
print("Precision train Score: \t", train_precision)
print("Precision test Score: \t", train_precision)
print("Recall train Score: \t\t", train_recall)
print("Recall test Score: \t\t", recall)
print("------------------------------------------------")
#print("Classification test report: ")
#print(metrics.classification_report(y_test_multi, y_pred_test_multi))


# In[34]:
"""
# grid search

print("grid search start...")
params = {"criterion":["entropy", "gini"], #"gini","entropy"
          #"min_samples_split": range(2,150),
          "min_samples_split": [13],
          #"min_samples_leaf": range(1,100),
          "min_samples_leaf": [42],
          "splitter":["best", "random"],
          "max_depth":[None],
          "min_weight_fraction_leaf":[0.0],
          #"max_features":["auto", "sqrt", "log2",None],
          "max_features":["sqrt", "log2",None],
          "random_state":[None],
          "max_leaf_nodes":[None],
          "min_impurity_decrease":[0.0],
          "class_weight":["balanced",None],
          "ccp_alpha":[0.0],
          }

clf = DecisionTreeClassifier()
gs = GridSearchCV(estimator=clf, param_grid=params,
                n_jobs=-1, cv=5, error_score='raise')
gs.fit(X_train_multi, y_train_multi)


gs.best_estimator_.fit(X_train_multi, y_train_multi)
y_pred_train_multi = gs.best_estimator_.predict(X_train_multi)
y_pred_test_multi = gs.best_estimator_.predict(X_test_multi)
y_true = y_test_multi


print(gs.best_params_)
print(gs.best_estimator_)
print(gs.best_score_)
print("Accuracy train", metrics.accuracy_score(y_train_multi, y_pred_train_multi))
print("Accuracy test", metrics.accuracy_score(y_test_multi, y_pred_test_multi))
f1_weighted_train = round(metrics.f1_score(y_train_multi, y_pred_train_multi, average='weighted'), 5)
f1_weighted = round(metrics.f1_score(y_test_multi, y_pred_test_multi, average='weighted'), 5)
print("test_f1",f1_weighted_train)
print("test_f1",f1_weighted)

# In[11]:
"""

#last model
""""
clf_ = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=19, min_samples_split=10, max_features=45, splitter='best',ccp_alpha=0.0000002)
clf_.fit(X_train_multi,y_train_multi)
y_train_pred_multi = clf_.predict(X_train_multi)
y_test_pred_multi = clf_.predict(X_test_multi)


accuracy = round(metrics.accuracy_score(y_test_multi, y_test_pred_multi), 5)
f1_weighted = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
f1_micro = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
f1_macro = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
precision_weighted = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
precision_micro = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
precision_macro = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
recall_weighted = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
recall_micro = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
recall_macro = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='macro'), 5)

print("Accuracy Score: \t", accuracy)
print("F1 weighted Score: \t\t", f1_weighted)
print("F1 micro Score: \t\t", f1_micro)
print("F1 macro Score: \t\t", f1_macro)
print("Precision weighted Score: \t", precision_weighted)
print("Precision micro Score: \t", precision_micro)
print("Precision macro Score: \t", precision_macro)
print("Recall Score: \t\t", recall_weighted)
print("Recall micro Score: \t\t", recall_micro)
print("Recall macro Score: \t\t", recall_macro)
print("------------------------------------------------")
print("Classification report: ")
print(metrics.classification_report(y_test_multi, y_test_pred_multi))
"""

elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")