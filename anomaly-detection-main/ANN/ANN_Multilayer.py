#!/usr/bin/env python
# coding: utf-8

# In[1]:
# to remove tensorfow message
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import numpy as np
import pandas as pd

sys.path.append("..")
from Functions.UNSW_DF import DF_XY

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras_visualizer import visualizer
from ann_visualizer.visualize import ann_viz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

x_train, x_test, y_train, y_test = DF_XY()


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
    precision = precision_score(dataset, yhat_classes, zero_division=0)

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
    recall = recall_score(dataset, yhat_classes, zero_division=0)

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
    f1 = f1_score(dataset, yhat_classes)

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
ann_train_recall, ann_test_recall = [], []
# Empty lists' for F1
ann_train_f1, ann_test_f1 = [], []
# Empty lists' for error rate
ann_train_error, ann_test_error = [], []
# Empty lists' for attributes
ann_epochs, ann_no_of_layers, ann_runtime = [], [], []


def ANN_predict(epoch_start, epoch_end, epoch_step, batch_size):
    """Predicts an ANN model with a predefined model.
    Args:
        epoch_start (int): Start value for Epoch
        epoch_end (int): End value for Epoch
        bactch_size (int): Keras batch size
    """
    for epoch in range(epoch_start, epoch_end + 1, epoch_step):
        # define the keras model
        model = Sequential()
        model.add(Dense(100, input_dim=x_train.shape[1], activation='relu'))
        model.add(Dense(75, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        predict_x = model.predict(x_test)

        # Start time for calculating the runtime for each epoch
        start_time = time.time()
        print(f"\n## ---------- EPOCH {epoch} ----------- ##\n")
        # fit the keras model on the dataset
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size)

        ###------------------------###
        ###--- MODEL PREDICTION ---###
        ###------------------------###
        # predict probabilities for train set
        yhat_probs_train = model.predict(x_train, verbose=0)
        # predict crisp classes for train set
        yhat_classes_train = np.argmax(yhat_probs_train, axis=1)

        # predict probabilities for test set
        yhat_probs_test = model.predict(x_test, verbose=0)
        # predict crisp classes for test set
        yhat_classes_test = np.argmax(yhat_probs_test, axis=1)

        ###--------------------###
        ###--- MODEL SCORES ---###
        ###--------------------###

        ###--- ACCURACY & ERROR ---###
        train_accuracy, train_error = get_accuracy_error(y_train, yhat_classes_train, "train")
        test_accuracy, test_error = get_accuracy_error(y_test, yhat_classes_test, "test")

        ###--- PRECISION ---###
        train_precision = get_precision(y_train, yhat_classes_train, "train")
        test_precision = get_precision(y_test, yhat_classes_test, "test")

        ###--- RECALL ---###
        train_recall = get_recall(y_train, yhat_classes_train, "train")
        test_recall = get_recall(y_test, yhat_classes_test, "test")

        ###--- F1 ---###
        train_f1 = get_F1(y_train, yhat_classes_train, "train")
        test_f1 = get_F1(y_test, yhat_classes_test, "test")

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
    return model


# In[4]:


model = ANN_predict(epoch_start=0, epoch_end=150, epoch_step=10, batch_size=10)

# # Neural Network

# In[5]:


# dictionary of lists
dict = {'epochs': ann_epochs,
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

df = pd.DataFrame(dict)
# EXPORT AS CSV when done.
df.to_csv('ANN_200E_4L(500-250-100-1)_AF(Relu-Sigmoid).csv', index=False)
df

# In[ ]:


# UNSW_barplot(data=df, to_range=29 ,x_label="epochs", y_label="Score", title="ANN", x_size=30, y_size=8)


# In[ ]:


# ann_viz(model, title="Neural Network")


# In[ ]:


# visualizer(model, format='png', view=True)


# In[ ]:


# In[ ]:


# evaluate the keras model for training and testing data
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy for training\t: %.2f' % (train_acc * 100))
print('Accuracy for testing\t: %.2f' % (test_acc * 100))
y_pred_test = model.predict(x_test)
# Calculate precision for each class
precision_per_class = precision_score(y_test, y_pred_test, average=None)

# Display the precision for each class
for class_idx, precision in enumerate(precision_per_class):
    print(f"Precision for class {class_idx}: {precision:.2f}")
