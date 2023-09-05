#!/usr/bin/env python
# coding: utf-8
from pprint import pprint
import sklearn
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
import sys
sys.path.append("..")
from Functions.UNSW_DF import *

# In[ ]:
X_train, X_test, y_train, y_test = DF_XY()

# importing Dataset
train, test = DF_preprocessed_traintest()


#test = pd.read_csv("../Dataset/test_pp3.csv", sep=',', header=0)
#train = pd.read_csv("../Anomaly-Detection-main/Dataset/train_pp3.csv", sep=',', header=0)


# In[ ]:
X_train = train.drop(['label'], axis=1)
X_test = test.drop(['label'], axis=1)
y_train = train.loc[:, ['label']]
y_test = test.loc[:, ['label']]


cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit((X_train,y_train)
y_pred = cls.predict(X_test)


print("Accuracy test", sklearn.metrics.accuracy_score(y_test, y_pred))

