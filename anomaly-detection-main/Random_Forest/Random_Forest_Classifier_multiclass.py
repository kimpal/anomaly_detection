#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier Multiclass

import time
import pandas as pd
import numpy as np
import sys

from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.append("..")
from Functions.UNSW_DF import DF_XY_MULTI
from sklearn import metrics
# importing random forest classifier from assemble module
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

x_train_multi, x_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()

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

#"""
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
model.fit(x_train_multi, y_train_multi)

# performing predictions on the training and test dataset
y_pred_train = model.predict(x_train_multi)
y_pred_test = model.predict(x_test_multi)


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
#"""

"""
# Random forest baseline with no feature selection
# fit the model
print(X_train_multi.shape)
RF_model = RandomForestClassifier(n_estimators=16, max_depth=17, criterion="entropy", bootstrap=True, min_samples_split=2, min_samples_leaf=1)
RF_model.fit(X_train_multi, y_train_multi)
print(RF_model)
# evaluate the model
y_pred_train = RF_model.predict(X_train_multi)
y_pred_test = RF_model.predict(X_test_multi)
train_accuracy = round(accuracy_score(y_train_multi, y_pred_train),5)
test_accuracy = round(accuracy_score(y_test_multi, y_pred_test),5)
print(f"Training accuracy: {train_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# classification report code
print(metrics.classification_report(y_test_multi, y_pred_test))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, y_pred_test)
print("confusion matrix:")
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
"""

"""
# evaluation random forest model using threshold=0.006 Feature chose whit different models
# feature selection
def select_features(x_train, y_train, x_test):
	# configure to select a subset of features whit rf
	fs = SelectFromModel(RandomForestClassifier(n_estimators=16), threshold=0.006)
	# configure to select a subset of features whit XGBOoST
	#fs = SelectFromModel(XGBClassifier(), threshold=0.006)
	# configure to select a subset of feature whit Decision Tree feature
	#fs = SelectFromModel(DecisionTreeClassifier(), threshold=0.006)
	# learn relationship from training data
	fs.fit(x_train, y_train)
	# transform train input data
	x_train_fs = fs.transform(x_train)
	# transform test input data
	x_test_fs = fs.transform(x_test)
	return x_train_fs, x_test_fs, fs
# feature selection
x_train_fs, x_test_fs, fs = select_features(X_train_multi, y_train_multi, X_test_multi)
print(x_train_fs.shape)
# fit the model
RF_model = RandomForestClassifier(n_estimators=16, max_depth=17, criterion="entropy", bootstrap=True, min_samples_split=2, min_samples_leaf=1)
RF_model.fit(x_train_fs, y_train_multi)
print(RF_model)
# evaluate the model
y_pred_train = RF_model.predict(x_train_fs)
y_pred_test = RF_model.predict(x_test_fs)
train_accuracy = round(accuracy_score(y_train_multi, y_pred_train),5)
test_accuracy = round(accuracy_score(y_test_multi, y_pred_test),5)
print(f"Training accuracy: {train_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# classification report code
print(metrics.classification_report(y_test_multi, y_pred_test))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, y_pred_test)
print("confusion matrix:")
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
"""

elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")
