#!/usr/bin/env python
# coding: utf-8

# In[18]:
import pandas as pd
import numpy as np
from matplotlib import pyplot
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
