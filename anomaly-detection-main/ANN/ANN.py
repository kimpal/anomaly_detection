#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import pandas as pd
import time
import sys
sys.path.append("..")
# from Functions.UNSW_DF_Multi_onehot import DF_XY_Multi
# from Functions.UNSW_DF_Multi_label import DF_XY_Multi


# from numpy import loadtxt
# from keras_visualizer import visualizer
# from ann_visualizer.visualize import ann_viz

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

# My Model
# 22: 85.79, 83.32
# 42: 

# Sklearn
# MLP: 83.9, 83.1

# Paper 19F: 79.46, 77.51
# Paper 42F: 79.91, 75.62
FEATURES = "42FE"  #42FE, 22FE

x_train = pd.read_csv(f'../Used/xtrain_{FEATURES}.csv')
y_train = pd.read_csv(f'../Used/ytrain_{FEATURES}.csv')

x_test = pd.read_csv(f'../Used/xtest_{FEATURES}.csv')
y_test = pd.read_csv(f'../Used/ytest_{FEATURES}.csv')

x_train = scaler1.fit_transform(x_train)
x_test = scaler2.fit_transform(x_test)

print(x_train.shape)
print(x_test.shape)

# relu, : 82.3, 77.5

runs = 10
run2 = 10

lr = 0.002
na = 15

act1 = 'relu'
epoch = 20

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
        print('Error for training:\t %f' % train_error)
        return train_accuracy, train_error
    elif train_test_string == "test":
        # Test accuracy
        test_accuracy = accuracy
        print(f'Accuracy for testing:\t {test_accuracy}')
        # Error rate for test
        test_error = 1 - test_accuracy
        print('Error for testing:\t %f' % test_error)
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
    precision = precision_score(
        dataset, yhat_classes, zero_division=0, average='weighted')

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
    recall = recall_score(dataset, yhat_classes,
                          zero_division=0, average='weighted')

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
        print('F1 for training:\t %f' % train_f1)
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
ann_train_recall, ann_test_recall = [], []
# Empty lists' for F1
ann_train_f1, ann_test_f1 = [], []
# Empty lists' for error rate
ann_train_error, ann_test_error = [], []
# Empty lists' for attributes
ann_epochs, ann_no_of_layers, ann_runtime = [], [], []

def ANN_predict(e_start,e_end, batch_size):
    """Predicts an ANN model with a predefined model.
    Args:
        epoch_start (int): Start value for Epoch
        epoch_end (int): End value for Epoch
        bactch_size (int): Keras batch size
    """
    for x in range(0, runs):
        for epochs in range(e_start, e_end+1):
            # define the keras model
             model = Sequential()
             model.add(
                 Dense(na, input_dim=x_train.shape[1], activation=act1))
             model.add(Dense(len(np.unique(y_train)), activation='softmax'))
             # compile the keras model
             model.compile(loss='sparse_categorical_crossentropy',
                 optimizer=tf.keras.optimizers.Adam(learning_rate = lr), metrics=['accuracy'])
             # predict_x = model.predict(x_test)
             # Start time for calculating the runtime for each epoch
             start_time = time.time()
             print(f"\n## ---------- EPOCH {epoch} ----------- ##\n")
             # fit the keras model on the dataset
             model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                       shuffle=True, validation_split=0.25)
             ### ------------------------###
             ### --- MODEL PREDICTION ---###
             ### ------------------------###
             # predict probabilities for train set
             yhat_probs_train = (model.predict(x_train))
             # predict crisp classes for train set
             yhat_classes_train = np.argmax(yhat_probs_train, axis=1)
             # predict probabilities for test set
             yhat_probs_test = np.round(model.predict(x_test, verbose=0))
             # predict crisp classes for test set
             yhat_classes_test = np.argmax(yhat_probs_test, axis=1)
             ### --------------------###
             ### --- MODEL SCORES ---###
             ### --------------------###
             # Label
             used_train = yhat_classes_train
             used_test = yhat_classes_test
             ### --- ACCURACY & ERROR ---###
             train_accuracy, train_error = get_accuracy_error(
                 y_train, used_train, "train")
             test_accuracy, test_error = get_accuracy_error(
                 y_test, used_test, "test")
             ### --- PRECISION ---###
             train_precision = get_precision(y_train, used_train, "train")
             test_precision = get_precision(y_test, used_test, "test")
             ### --- RECALL ---###
             train_recall = get_recall(y_train, used_train, "train")
             test_recall = get_recall(y_test, used_test, "test")
             ### --- F1 ---###
             train_f1 = get_F1(y_train, used_train, "train")
             test_f1 = get_F1(y_test, used_test, "test")
             # Appending Scores to lists'
             ann_train_accuracy.append(train_accuracy)
             ann_test_accuracy.append(test_accuracy)

             ann_train_error.append(train_error)
             ann_test_error.append(test_error)

             ann_train_precision.append(train_precision)
             ann_test_precision.append(test_precision)

             ann_train_recall.append(train_recall)
             ann_test_recall.append(test_recall)

             ann_train_f1.append(train_f1)
             ann_test_f1.append(test_f1)

             # Appending attributes
             ann_no_of_layers.append(len(model.layers))
             ann_epochs.append(epoch)
             elapsed_time = round((time.time() - start_time), 3)
             ann_runtime.append(elapsed_time)
             print(f"Runtime for Epoch {epoch}:\t {elapsed_time}s")
    print("Train mean val: ", np.mean(ann_train_accuracy))
    print("Test mean val: ", np.mean(ann_test_accuracy))
    return model

# In[4]:
print("ANN start")
model = ANN_predict(e_start= epoch,e_end=epoch, batch_size=128)

# # Neural Network
y_t = model.predict(x_test)
y_te = model.predict(x_train)

y_t = np.argmax(y_t, axis=1)
y_te = np.argmax(y_te, axis=1)

print("MLP start")
clf = MLPClassifier(hidden_layer_sizes=(15,),random_state=1, max_iter=20, learning_rate_init = lr, learning_rate = 'adaptive').fit(x_train, np.ravel(y_train))
print("\nMLP train: ", clf.score(x_train, np.ravel(y_train)))
print("MLP test: ", clf.score(x_test, np.ravel(y_test)))

# train_acc = round(accuracy_score(y_train, y_te),5)
# test_acc = round(accuracy_score(y_test, y_t),5)

# print("train ", train_acc)
# print("test ", test_acc)

# In[5]:

# evaluate the keras model for training and testing data
a_a = []
a_b = []
for x in range(0,run2):
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    a_a.append(train_acc*100)
    a_b.append(test_acc*100)
print('\nANN average training\t: %.2f' % np.mean(a_a))
print('ANN average testing\t: %.2f' % np.mean(a_b))

print("X train: ", x_train.shape)
print("X test: ", x_test.shape)

# In[ ]:
# dictionary of lists
thisdict = {'epochs': ann_epochs,
        'no_layers': ann_no_of_layers,
        'accuracy_train': ann_train_accuracy,
        'accuracy_test': ann_test_accuracy,
        'error_train': ann_train_error,
        'error_test': ann_test_error,
        'precision_train': ann_train_precision,
        'precision_test': ann_test_precision,
            'F1_train': ann_train_f1,
        'F1_test': ann_test_f1,
            'recall_train': ann_train_recall,
            'recall_test': ann_test_recall,
            'runtime(s)': ann_runtime
            }

df = pd.DataFrame(thisdict)
# EXPORT AS CSV when done.
df.to_csv(f'ANN_Flip_20E_(150)_AF(Relu)_{FEATURES}_.csv', index=False)

df

# In[ ]:
# UNSW_barplot(data=df, to_range=29 ,x_label="epochs", y_label="Score", title="ANN", x_size=30, y_size=8)
