#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier Multiclass

import time
import pandas as pd
import numpy as np
import sys

from matplotlib import pyplot
from sklearn.metrics import accuracy_score

sys.path.append("..")
from Functions.UNSW_DF import DF_XY_MULTI
from sklearn import metrics
# importing random forest classifier from assemble module
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

X_train_multi, X_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()

# importing Dataset
#train_multi, test_multi = DF_preprocessed_traintest_multi()


#dataframe = pd.merge(train_multi, test_multi)
#X = dataframe.drop(["attack_cat"], axis=1)
#y= dataframe["attack_cat"]

start_time = time.time()
"""
# ## Creating the classifier
start_time = time.time()
model = RandomForestClassifier()
# fit the model on the whole dataset
model.fit(X_train_multi, y_train_multi)

# performing predictions on the traing and test dataset
y_pred_train = model.predict(X_train_multi)
y_pred_test = model.predict(X_test_multi)
"""

#external code
"""
def print_scores(y_test_multi,y_preds_test,y_train_multi,y_preds_train):
#     fpr, tpr, thresholds = roc_curve(y_train, y_preds_train)
    # get the best threshold
#     J = tpr - fpr
#     ix = np.argmax(J)
#     best_thresh = thresholds[ix]
#     roc=roc_auc_score(y_true=y_test, y_score=y_preds_test)
#     print("Best threshold", best_thresh)
#     print("ROC-AUC", round(roc,4))
#     print("Precision", round(precision_score(y_true=y_test, y_pred=y_preds_test), 4))
#     print("Recall", round(recall_score(y_true=y_test, y_pred=y_preds_test), 4))
#     print("F1", round(f1_score(y_true=y_test, y_pred=y_preds_test), 4))
    print("accuracy", round(accuracy_score(y_true=y_test_multi, y_pred=y_preds_test), 4))

def xs_y(df_, targ):
    if not isinstance(targ, list):
        xs = df_[df_.columns.difference([targ])].copy()
    else:
        xs = df_[df_.columns.difference(targ)].copy()
    y = df_[targ].copy()
    return xs, y

X_train_multi, y_train_multi = xs_y(train_multi, targ='attack_cat')
X_test_multi, y_test_multi= xs_y(test_multi, targ='attack_cat')

clf=RandomForestClassifier(300)#doing this so calculating shapply values won't take me years
clf.fit(X_train_multi,y_train_multi)
pred=clf.predict(X_test_multi)
preds_train=clf.predict(X_train_multi)
print_scores(y_test_multi,pred,y_train_multi,preds_train)
"""

# test min_samples_split: 1.0
params = {"criterion": "entropy",
              "bootstrap":True,
              "n_estimators": 200,
              "max_depth":14,
              "min_samples_split":2,
              "min_samples_leaf": 1,
              "n_jobs": -1}


# define the model
model = RandomForestClassifier(**params)
model.set_params(**params)

# fit the model on the whole dataset
model.fit(X_train_multi, y_train_multi)

# performing predictions on the traing and test dataset
y_pred_train = model.predict(X_train_multi)
y_pred_test = model.predict(X_test_multi)


train_accuracy = accuracy_score(y_train_multi, y_pred_train)
test_accuracy = accuracy_score(y_test_multi, y_pred_test)
f1 = round(metrics.f1_score(y_test_multi, y_pred_test, average='weighted'), 5)
print(f"Training accuracy: \t{train_accuracy}\nTest accuracy: \t\t{test_accuracy}\nF1-score: \t\t{f1}\n")
print("------------------------------------------------------------------------------------------------")
#print(f"Training accuracy: \t{train_accuracy}\nTest accuracy: \t\t{test_accuracy}")
# Assuming y_train_multi, y_pred_train, y_test_multi, and y_pred_test are multi-class targets and predictions

train_accuracy = round(metrics.accuracy_score(y_train_multi, y_pred_train), 5)
test_accuracy = round(metrics.accuracy_score(y_test_multi, y_pred_test), 5)
macro_f1 = round(metrics.f1_score(y_test_multi, y_pred_test, average='macro'), 5)
micro_f1 = round(metrics.f1_score(y_test_multi, y_pred_test, average='micro'), 5)
weighted_f1 = round(metrics.f1_score(y_test_multi, y_pred_test, average='weighted'), 5)
macro_precision = round(metrics.precision_score(y_test_multi, y_pred_test, average='macro'), 5)
micro_precision = round(metrics.precision_score(y_test_multi, y_pred_test, average='micro'), 5)
weighted_precision = round(metrics.precision_score(y_test_multi, y_pred_test, average='weighted'), 5)
macro_recall = round(metrics.recall_score(y_test_multi, y_pred_test, average='macro'), 5)
micro_recall = round(metrics.recall_score(y_test_multi, y_pred_test, average='micro'), 5)
weighted_recall = round(metrics.recall_score(y_test_multi, y_pred_test, average='weighted'), 5)

print(f"Training accuracy: \t{train_accuracy}\nTest accuracy: \t\t{test_accuracy}")
print(f"Macro F1-score: \t{macro_f1}\nMicro F1-score: \t{micro_f1}\nWeighted F1-score: \t{weighted_f1}")
print(f"Macro precision: \t{macro_precision}\nMicro precision: \t{micro_precision}\nWeighted precision: \t{weighted_precision}")
print(f"Macro recall: \t\t{macro_recall}\nMicro recall: \t\t{micro_recall}\nWeighted recall: \t{weighted_recall}")
print("Classification report on test: ")
print(metrics.classification_report(y_test_multi, y_pred_test))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, y_pred_test)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.savefig('RFC_confusion_matrix.png')
pyplot.show()

elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")
