import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
  return 1/(1+np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
  return sigmoid(x) * (1- sigmoid(x))