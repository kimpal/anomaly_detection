#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import sys
sys.path.append("..")
from UNSW_DFv2 import *


# In[ ]:


reg_strength = 10000
learning_rate = 0.000001


# In[ ]:


train, test = DF_preprocessed_traintest()
X_train, X_test, y_train, y_test = DF_XY()


# In[ ]:


train.drop(train.index[10000:175341], axis=0, inplace=True)
test.drop(test.index[5000:82332], axis=0, inplace=True)
print(f"Train shape:\t {train.shape}\nTest shape:\t {test.shape}")


# In[ ]:


def initialize():
    print("training started ")
    W = stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))
    
    y_test_predicted = np.array([])
    for i in range(X_test.shape([0])):
        


# In[ ]:


def cost_computation(W, X, Y):
    #Calclate the hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    #equivalkent to max(0, distance)
    distances[distances < 0] = 0
    hinge_loss = reg_strength * (np.sum(distances) / N)
    #Calculate the cost
    cost = 1/2 *  np.dot(W, W) + hinge_loss
    return cost
    


# In[ ]:


def gradient_cost(W, X_batch, Y_batch):
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    
    dw = dw/len(Y_batch)
    return dw


# In[ ]:


def stochastic_gradient_descent(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01
    for epoch in range(1, max_epochs):
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = gradient_cost(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
    
    
    if epoch == 2 ** nth or epoch == max_epochs - 1:
        cost = cost_computation(weights, features, outputs)
        print("Epoch is: {} and cost is: {}".format(epoch, cost))
        if abs(prev_cost - cost) < cost_threshold * prev_cost:
            return weights
        prev_cost = cost
        nth += 1

    return weights

