import numpy as np

# Leaky Rectified Linear Unit (leaky ReLU) Activation Function
def leakyReLU(x):
  data = [max(0.05*value,value) for value in x]
  return np.array(data, dtype=float)

# Derivative for leaky ReLU 
def leakyReLU_derivative(x):
  data = [1 if value>0 else 0.05 for value in x]
  return np.array(data, dtype=float)