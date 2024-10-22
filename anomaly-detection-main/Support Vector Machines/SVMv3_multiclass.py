#!/usr/bin/env python
# coding: utf-8

# In[1]:
#import warnings
#warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import datetime
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import sys
sys.path.append("..")
from  Functions.UNSW_DF import DF_preprocessed_traintest_multi, DF_XY_MULTI

# time function
now = datetime.datetime.now()
print(now)
# In[2]:
start_time = time.time()


train_multi, test_multi = DF_preprocessed_traintest_multi()
X_train_multi, X_test_multi, y_train_multi, y_test_multi = DF_XY_MULTI()


# In[3]:
## external code whit multi class classification working pol
"""
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train_multi, y_train_multi)

poly_pred_train = poly.predict(X_train_multi)
poly_pred_test = poly.predict(X_test_multi)

# Train and test metrics on poly svm
poly_train_accuracy = accuracy_score(y_train_multi, poly_pred_train)
poly_train_f1 = f1_score(y_train_multi, poly_pred_train, average='weighted')
poly_test_accuracy = accuracy_score(y_test_multi, poly_pred_test)
poly_test_f1 = f1_score(y_test_multi, poly_pred_test, average='weighted')
# print('Train Accuracy (Polynomial Kernel): ', "%.2f" % (poly_train_accuracy*100))
# print('Train F1 (Polynomial Kernel): ', "%.2f" % (poly_train_f1*100))
# print('Test Accuracy (Polynomial Kernel): ', "%.2f" % (poly_test_accuracy*100))
# print('Test F1 (Polynomial Kernel): ', "%.2f" % (poly_test_f1*100))
print(f"Training accuracy: \t{poly_train_accuracy}\nTest accuracy: \t\t{poly_test_accuracy}\nWeighted F1-score: \t{poly_test_f1}")
print("Classification report on test: ")
print(metrics.classification_report(y_test_multi, poly_pred_test))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, poly_pred_test)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot();
pyplot.show()
"""

"""
# rbf kernel 
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train_multi, y_train_multi)

#rbf_pred = rbf.predict(X_test_multi)
rbf_pred_train = rbf.predict(X_train_multi)
rbf_pred_test = rbf.predict(X_test_multi)

#rbf train and test acc
rbf_train_accuracy = accuracy_score(y_train_multi, rbf_pred_train)
rbf_train_f1 = f1_score(y_train_multi, rbf_pred_train, average='weighted')
rbf_test_accuracy = accuracy_score(y_test_multi, rbf_pred_test)
rbf_test_f1 = f1_score(y_test_multi, rbf_pred_test, average='weighted')
#print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_train_accuracy*100))
#print('Train F1 (RBF Kernel): ', "%.2f" % (rbf_train_f1*100))
#print('Test Accuracy (RBF Kernel): ', "%.2f" % (rbf_test_accuracy*100))
#print('Test F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
print(f"Training accuracy:\t{rbf_train_accuracy*100}\nTest accuracy: \t\t{rbf_test_accuracy*100}\nWeighted F1-score: \t{rbf_test_f1*100}")
print("Classification report on test: ")
print(metrics.classification_report(y_test_multi, rbf_pred_test))
# Create the confusion matrix
cm = metrics.confusion_matrix(y_test_multi, rbf_pred_test)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.savefig('SVMv3_confusion_matrix.png')
pyplot.show()
"""



# In[4]:
"""
error = []
C_value = []
accuracy_score = []
precision_score = []
precision_score_micro = []
precision_score_macro = []
F1_score = []
F1_score_micro =[]
F1_score_macro =[]
recall_score = []
recall_score_micro = []
recall_score_weighted = []
recall_score_macro = []

def SVM_predict(c_start, c_end, svm_kernel, svm_gamma, svm_degree):
"""
"""
#    Predicts an SVM model with given arguments
#    Args:
#        c_start (int): C value start
#        c_end (int): C value end
#        svm_kernel (string): SVM kernel given in a string format: i.e. linear, poly, rbf
#        svm_degree (int): default=3, Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
"""
"""
    
    # c_start = c_start
    # c_end += 50
    print("starting SVM...")
    print(f"started: {now}")
    for c in range(c_start, c_end):
        SVM_model = SVC(kernel=svm_kernel, C = c, gamma=svm_gamma, degree=svm_degree)
        SVM_model.fit(X_train_multi, y_train_multi)
        pred_i = SVM_model.predict(X_test_multi)

        # Appending values to list
        error.append(np.mean(pred_i != y_test_multi))
        C_value.append(c)
        accuracy_score.append(metrics.accuracy_score(y_test_multi, pred_i))
        #precision_score.append(metrics.precision_score(y_test_multi, pred_i))
        #precision_score_weighted.append(metrics.precision_score(y_test_multi, pred_i, average='weighted' ))
        #precision_score_micro.append(metrics.precision_score(y_test_multi, pred_i, average='micro'))
        #precision_score_macro.append(metrics.precision_score(y_test_multi, pred_i, average='macro'))
        #F1_score_weighted.append(metrics.f1_score(y_test_multi, pred_i, average='weighted'))
        F1_score_micro.append(metrics.f1_score(y_test_multi, pred_i, average='micro'))
        F1_score_macro.append(metrics.f1_score(y_test_multi, pred_i, average='macro'))
        #F1_score.append(metrics.f1_score(y_test_multi, pred_i))
        #recall_score.append(metrics.recall_score(y_test_multi, pred_i))
        recall_score_weighted.append(metrics.recall_score(y_test_multi, pred_i, average='weighted'))
        recall_score_micro.append(metrics.recall_score(y_test_multi, pred_i, average='micro'))
        recall_score_macro.append(metrics.recall_score(y_test_multi, pred_i, average='macro'))
        print("in the running-----------")


        # Append values to lists
        error.append(np.mean(pred_i != y_test_multi))
        C_value.append(c)
        accuracy_score.append(metrics.accuracy_score(y_test_multi, pred_i, average='micro'))
        precision_score.append(metrics.precision_score(y_test_multi, pred_i, average='micro'))
        F1_score_micro.append(metrics.f1_score(y_test_multi, pred_i, average='micro'))
        #F1_score.append(metrics.f1_score(y_test_multi, pred_i, average='micro'))
        recall_score.append(metrics.recall_score(y_test_multi, pred_i, average='micro'))

c_start=1
c_end=11
svm_kernel="poly"
# Calling the function created above.
SVM_predict(c_start=c_start, c_end=c_end, svm_kernel=svm_kernel, svm_gamma="auto", svm_degree=3)


print("SVM has finished!")
print("creating CSV file...")
# Creating a dataframe and saving to file
# dictionary of lists
dict = {
        'C': C_value, 
        'accuracy': accuracy_score,
        #'precision': precision_score,
        'precision_micro': precision_score_micro,
        'precision_macro': precision_score_macro,
        #'F1': F1_score,
        'F1_micro': F1_score_micro,
        'F1_macro': F1_score_macro,
        #'recall': recall_score,
        'recall_micro': recall_score_micro,
        'recall_macro': recall_score_macro,
        'error': error
        }

df = pd.DataFrame(dict, index=C_value)
df.set_index("C", inplace = True)
df

#converting c_start and c_end to string to use in csv file name
c_start_s = str(c_start)
c_end_s = str(c_end)
svm_kernel_s = str(svm_kernel)

# EXPORT AS CSV when done.
df.to_csv('SVM_scores_multiclass(' + c_start_s + '-'+c_end_s +')_kernel_'+svm_kernel_s+'.csv')
"""

"""
# Feature extraction
# fit the model Baseline test whit out
SVM_model = svm.SVC(kernel='poly', degree=3, C=1)
SVM_model.fit(X_train_multi, y_train_multi)
print(SVM_model)
# evaluate the model
y_pred_train = SVM_model.predict(X_train_multi)
y_pred_test = SVM_model.predict(X_test_multi)
train_accuracy = round(accuracy_score(y_train_multi, y_pred_train),5)
test_accuracy = round(accuracy_score(y_test_multi, y_pred_test),5)
print(f"Training accuracy: {train_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# classification report code
print(classification_report(y_test_multi, y_pred_test))
# Create the confusion matrix
cm = confusion_matrix(y_test_multi, y_pred_test)
print("confusion matrix:")
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
"""

#"""
# evaluation of SVM tre using features whit threshold=0.006 on diferent feature selection metods
# feature selection
def select_features(x_train, y_train, x_test):
	# configure to select a subset of features whit rf
	#fs = SelectFromModel(RandomForestClassifier(n_estimators=16), threshold=0.006) #max_features=50 )
	# configure to select a subset of features whit XGBOOST
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
SVM_model = svm.SVC(kernel='poly', degree=3, C=1)
SVM_model.fit(x_train_fs, y_train_multi)
print(SVM_model)
# evaluate the model
y_pred_train = SVM_model.predict(x_train_fs)
y_pred_test = SVM_model.predict(x_test_fs)
train_accuracy = round(accuracy_score(y_train_multi, y_pred_train),5)
test_accuracy = round(accuracy_score(y_test_multi, y_pred_test),5)
print(f"Training accuracy: {train_accuracy*100}\nTest accuracy: \t {test_accuracy*100}")
# classification report code
print(classification_report(y_test_multi, y_pred_test))
# Create the confusion matrix
cm = confusion_matrix(y_test_multi, y_pred_test)
print("confusion matrix:")
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
pyplot.show()
#"""

elapsed_time = round((time.time() - start_time), 3)
print(f"Runtime: {elapsed_time}s")
print("code is finished..")
