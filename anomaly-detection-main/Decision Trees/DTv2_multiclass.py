#!/usr/bin/env python
# coding: utf-8

# In[1]:
from matplotlib import pyplot
import time
import sys

from sklearn.feature_selection import SelectFromModel

sys.path.append("..")
from Functions.UNSW_DF import DF_XY_MULTI, DF_preprocessed_traintest_multi
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import metrics
x_train_multi, x_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()

X_train_multi, X_test_multi, y_train_multi, y_test_multi = x_train_multi, x_test_multi, y_train_multi, y_test_multi
# importing Dataset
#train_multi, test_multi = DF_preprocessed_traintest_multi()
# In[2]:

start_time = time.time()

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
print("F1 test Score: \t\t\t", f1)
print("Precision test Score: \t", precision_train)
print("Precision test Score: \t", precision)
print("Recall test Score: \t\t", recall)
print("------------------------------------------------")
print("Classification test report: ")
print(metrics.classification_report(y_test_multi, y_pred_test_multi))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, y_pred_test_multi)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot();
pyplot.show()
"""



# In[9]:

"""
print("------------------------------------")
print("next model starts...")
# second model
clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, min_samples_split=13, max_features=45 ,ccp_alpha=0.0) #splitter="best")  #ccp_alpha=0.0000002) # test whit ccp_alpha=0.0
clf = clf.fit(x_train_multi, y_train_multi)
y_pred_train_multi = clf.predict(x_train_multi)
y_pred_test_multi = clf.predict(x_test_multi)

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
print("F1 test Score: \t\t\t", f1)
print("Precision train Score: \t", train_precision)
print("Precision test Score: \t", train_precision)
print("Recall train Score: \t", train_recall)
print("Recall test Score: \t\t", recall)
print("------------------------------------------------")
print("Classification test report: ")
print(metrics.classification_report(y_test_multi, y_pred_test_multi))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, y_pred_test_multi)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot();
pyplot.show()
"""
# In[34]:
"""
# grid search

print("grid search start...")
params = {"criterion":["entropy", "gini"], #"gini","entropy"
          "min_samples_split": range(2,150),
          #"min_samples_split": [13],
          "min_samples_leaf": range(1,100),
          #"min_samples_leaf": [42],
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
gs.fit(x_train_multi, y_train_multi)


gs.best_estimator_.fit(x_train_multi, y_train_multi)
y_train_pred_multi = gs.best_estimator_.predict(x_train_multi)
y_test_pred_multi = gs.best_estimator_.predict(x_test_multi)
y_true = y_test_multi


print(gs.best_params_)
print(gs.best_estimator_)
print(gs.best_score_)
# train and test accuracy, f1, precision, recall and classification report
accuracy_train = round(metrics.accuracy_score(y_train_multi, y_train_pred_multi), 5)
accuracy_test = round(metrics.accuracy_score(y_test_multi, y_test_pred_multi), 5)
f1_weighted = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
f1_micro = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
f1_macro = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
precision_weighted = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
precision_micro = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
precision_macro = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
recall_weighted = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
recall_micro = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
recall_macro = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
print(f"F1 weighted Score: \t{f1_weighted}\nF1 micro Score: \t{f1_micro}\nF1 macro Score: \t\t{f1_macro}")
print("Precision weighted Score: \t", precision_weighted)
print(f"Precision micro Score: \t{precision_micro}\nPrecision macro Score: \t{precision_macro} ")
print(f"Recall Score: \t{recall_weighted}\nRecall micro Score: \t{recall_micro}\nRecall macro Score: \t{recall_macro}")
print("------------------------------------------------")
print(f"Training accuracy score: \t{accuracy_train}\nTest accuracy score: \t\t{accuracy_test}")
print("Classification report: ")
print(metrics.classification_report(y_test_multi, y_test_pred_multi))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, y_test_pred_multi)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot();
pyplot.show()
# In[11]:
"""

#last model
"""
clf_ = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=19, min_samples_split=10, max_features=45, splitter='best',ccp_alpha=0.0000002)
clf_.fit(X_train_multi,y_train_multi)
y_train_pred_multi = clf_.predict(X_train_multi)
y_test_pred_multi = clf_.predict(X_test_multi)

# train and test accuracy, f1, precision, recall and classification report
accuracy_train = round(metrics.accuracy_score(y_train_multi, y_train_pred_multi), 5)
accuracy_test = round(metrics.accuracy_score(y_test_multi, y_test_pred_multi), 5)
f1_weighted = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
f1_micro = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
f1_macro = round(metrics.f1_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
precision_weighted = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
precision_micro = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
precision_macro = round(metrics.precision_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
recall_weighted = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='weighted'), 5)
recall_micro = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='micro'), 5)
recall_macro = round(metrics.recall_score(y_test_multi, y_test_pred_multi, average='macro'), 5)
print(f"F1 weighted Score: \t{f1_weighted}\nF1 micro Score: \t{f1_micro}\nF1 macro Score: \t\t{f1_macro}")
print("Precision weighted Score: \t", precision_weighted)
print(f"Precision micro Score: \t{precision_micro}\nPrecision macro Score: \t{precision_macro} ")
print(f"Recall Score: \t{recall_weighted}\nRecall micro Score: \t{recall_micro}\nRecall macro Score: \t{recall_macro}")
print("------------------------------------------------")
print(f"Training accuracy score: \t{accuracy_train}\nTest accuracy score: \t\t{accuracy_test}")
print("Classification report: ")
print(metrics.classification_report(y_test_multi, y_test_pred_multi))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, y_test_pred_multi)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot();
pyplot.show()
"""

#"""
#Bace line test on Decision Tree Before test on feature selection
print(X_train_multi.shape)
# fit the model
DT_model = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, min_samples_split=13, max_features=45 ,ccp_alpha=0.0)
DT_model.fit(X_train_multi, y_train_multi)
# evaluate the model
print(DT_model)
y_pred_train = DT_model.predict(X_train_multi)
y_pred_test = DT_model.predict(X_test_multi)
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
#"""


"""
# evaluation of Decision tre using features whit threshold=0.006 on different feature selection methods
# feature selection
def select_features(x_train, y_train, x_test):
	# configure to select a subset of features whit rf
	#fs = SelectFromModel(RandomForestClassifier(n_estimators=16), threshold=0.006) #max_features=50 )
	# configure to select a subset of features whit XGBOoST
	#fs = SelectFromModel(XGBClassifier(), threshold=0.006) #max_features=50 )
	# configure to select a subset of feature whit Decision Tree feature
	fs = SelectFromModel(DecisionTreeClassifier(), threshold=0.006) #max_features=50 ) DecisionTreeClassifier
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
DT_model = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20, min_samples_split=13, max_features=45 ,ccp_alpha=0.0)
DT_model.fit(x_train_fs, y_train_multi)
# evaluate the model
print(DT_model)
y_pred_train = DT_model.predict(x_train_fs)
y_pred_test = DT_model.predict(x_test_fs)
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

# end time of the model
elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")