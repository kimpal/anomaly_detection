#!/usr/bin/env python
# coding: utf-8

# In[18]:
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

train = pd.read_csv("../Dataset/train_pp3_multi.csv")
test = pd.read_csv("../Dataset/test_pp3_multi.csv")

x_train, y_train = train.drop(["attack_cat"], axis=1), train["attack_cat"]
x_test, y_test = test.drop(["attack_cat"], axis=1), test["attack_cat"]
print('X_train Shape: ', '\t', x_train.shape)
print('y_train Shape: ', '\t', y_train.shape)
print('X_test Shape: ', '\t\t', x_test.shape)
print('y_test Shape: ', '\t\t', y_test.shape)

start_time = time.time()

"""
model = XGBClassifier()
model.fit(x_train, y_train)
print(model)

# evaluate the model
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)
train_accuracy = round(accuracy_score(y_train, y_pred_train),5)
test_accuracy = round(accuracy_score(y_test, y_pred_test),)
print(f"Training accuracy: {train_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# clasification report code
print(classification_report(y_test, y_pred_test))
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("confusion matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot();
pyplot.savefig('XGBoost_confusion_matrix.png')
pyplot.show()
"""


"""
# Baseline test on XGBoost Before feature selection
print(x_train.shape)
# fit the model
XGBC_model = XGBClassifier()
XGBC_model.fit(x_train, y_train)
# evaluate the model
print(XGBC_model)
y_pred_train = XGBC_model.predict(x_train)
y_pred_test = XGBC_model.predict(x_test)
train_accuracy = round(accuracy_score(y_train, y_pred_train),5)
test_accuracy = round(accuracy_score(y_test, y_pred_test),5)
print(f"Training accuracy: {train_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# clasification report code
print(classification_report(y_test, y_pred_test))
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("confusion matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
"""

#"""
# evaluation of XGBOOST tre using features whit threshold=0.006 on different feature selection methods
# feature selection
def select_features(x_train, y_train, x_test):
	# configure to select a subset of features whit rf
	fs = SelectFromModel(RandomForestClassifier(n_estimators=16), threshold=0.006) #max_features=50 )
	# configure to select a subset of features whit XGBOOST
	#fs = SelectFromModel(XGBClassifier(), threshold=0.006) #max_features=50 )
	# configure to select a subset of features whit Decision Tree feature
	#fs = SelectFromModel(DecisionTreeClassifier(), threshold=0.006) #max_features=50 ) DecisionTreeClassifier
	# learn relationship from training data
	fs.fit(x_train, y_train)
	# transform train input data
	x_train_fs = fs.transform(x_train)
	# transform test input data
	x_test_fs = fs.transform(x_test)
	return x_train_fs, x_test_fs, fs
# feature selection
x_train_fs, x_test_fs, fs = select_features(x_train, y_train, x_test)
print(x_train_fs.shape)
# fit the model
XGBC_model = XGBClassifier()
XGBC_model.fit(x_train_fs, y_train)
# evaluate the model
print(XGBC_model)
y_pred_train = XGBC_model.predict(x_train_fs)
y_pred_test = XGBC_model.predict(x_test_fs)
train_accuracy = round(accuracy_score(y_train, y_pred_train),5)
test_accuracy = round(accuracy_score(y_test, y_pred_test),5)
print(f"Training accuracy: {train_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# classification report code
print(classification_report(y_test, y_pred_test))
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("confusion matrix:")
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
#"""

elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")
print("code is finished..")