#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.append("..")
import time

from Functions.UNSW_DF_Multi_label import DF_XY_Multi


import pandas as pd
import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras_visualizer import visualizer
from ann_visualizer.visualize import ann_viz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

x_train, x_test, y_train, y_test = DF_XY_Multi()

# In[2]:


# ACCURACY
def get_accuracy_error(dataset, yhat_classes, train_test_string):
    """Get accuracy and error rate from given keras model.

    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output

    Returns:
        float: returns accuracy and error rate for given model and datasets.
    """
    accuracy = accuracy_score(dataset, yhat_classes)
    
    if train_test_string == "train":
        # Train accuracy
        train_accuracy = accuracy
        print(f'Accuracy for training:\t {train_accuracy}')
        # Error rate for train
        train_error = 1 - train_accuracy
        print('Error for training:\t %f' %train_error)
        return train_accuracy, train_error
    elif train_test_string == "test":
        # Test accuracy
        test_accuracy = accuracy
        print(f'Accuracy for testing:\t {test_accuracy}')
        # Error rate for test
        test_error = 1 - test_accuracy
        print('Error for testing:\t %f' %test_error)
        return test_accuracy, test_error
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")

# PRECISION
def get_precision(dataset, yhat_classes, train_test_string):
    """Get precision for given keras model: Precision = tp / (tp + fp)
    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output

    Returns:
        float:  Returns precision for given model and dataset
    """ 
    
    precision = precision_score(dataset, yhat_classes, zero_division=0, average='weighted')
    
    if train_test_string == "train":
        train_precision = precision
        print('Precision for training:\t %f' % train_precision)
        return train_precision
    elif train_test_string == "test":
        test_precision = precision
        print('Precision for testing:\t %f' % test_precision)
        return test_precision
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")

# RECALL
def get_recall(dataset, yhat_classes, train_test_string):
    """Get Recall for given keras model: Recall = tp / (tp + fn)
    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output

    Returns:
        float:  Returns recall for given model and dataset
    """
    recall = recall_score(dataset, yhat_classes, zero_division=0, average='weighted')

    
    if train_test_string == "train":
        train_recall = recall
        print('Recall for training:\t %f' % train_recall)
        return train_recall
    elif train_test_string == "test":
        test_recall = recall
        print('Recall for testing:\t %f' % test_recall)
        return test_recall
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")
        
# F1
def get_F1(dataset, yhat_classes, train_test_string):
    """Get F1 for given keras model: F1 = 2 tp / (2 tp + fp + fn)
    Args:
        dataset (dataframe): Dataframe for the model, either y_train or y_test
        yhat_classes (prediction): model prediction for crisp classes
        train_test_string (string): Represent string either as "train" or "test" for correct output

    Returns:
        float:  Returns F1 for given model and dataset
    """
    f1 = f1_score(dataset, yhat_classes, average='weighted')
    
    if train_test_string == "train":
        train_f1 = f1
        print('F1 for training:\t\t %f' % train_f1)
        return train_f1
    elif train_test_string == "test":
        test_f1 = f1
        print('F1 for testing:\t\t %f' % test_f1)
        return test_f1
    else:
        print("Type either 'train' or 'test' for train_test_string parameter")


# In[3]:


# Empty lists' for accuracy
ann_train_accuracy, ann_test_accuracy = [], []
# Empty lists' for precision
ann_train_precision, ann_test_precision = [], []
# Empty lists' for accuracy
ann_train_recall, ann_test_recall= [], []
# Empty lists' for F1
ann_train_f1, ann_test_f1= [], []
# Empty lists' for error rate
ann_train_error, ann_test_error = [], []
# Empty lists' for attributes
ann_epochs, ann_no_of_layers, ann_runtime= [], [], []

def ANN_predict(activation = 'relu', units = 10):
    """Predicts an ANN model with a predefined model.

    Args:
        epoch_start (int): Start value for Epoch
        epoch_end (int): End value for Epoch
        bactch_size (int): Keras batch size
    """
    # Best: 0.763233 using {'activation': 'softsign', 'optimizer': 'adam', 'units': 200}
    # Best: 0.772980 2 layers
    # Best: Best: 0.771970 using {'activation': 'elu', 'units': 500} 3 layers
    # define the keras model
    model = Sequential()
    model.add(Dense(200, input_dim=x_train.shape[1], activation= 'softsign'))
    model.add(Dense(units, activation= activation))
    model.add(Dense(len(y_train.unique()), activation= 'sigmoid'))

    # compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
    
    return model

# In[]:
from scikeras.wrappers import KerasClassifier

#batch_sizes = [10, 20, 40, 60, 80, 100]
units11 = [500, 400, 300, 200, 100, 50, 20]
#epochs = [10, 50, 100]
activation11 = ['relu', 'sigmoid','softmax','softplus','softsign','tanh','selu','elu', 'exponential']
#opti = ['adam', 'rmsprop','sgd','adamw','adadelta','adamax','adafactor','nadam','ftrl']
    
model = KerasClassifier(model=ANN_predict, epochs = 20, batch_size = 100, units = units11, activation = activation11, verbose=0)
    
# In[]:
from sklearn.model_selection import GridSearchCV



param_grid = dict(units = units11, activation = activation11)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='accuracy')
grid_result = grid.fit(x_train, y_train)

# In[]:

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# In[]:

df = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    
# EXPORT AS CSV when done.
df.to_csv('ANN_Multi_gridsearch_10_classes_1_Layer_units_act_opti.csv')        



    