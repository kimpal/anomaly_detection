import numpy as np

# Rectified Linear Unit (ReLU)
def ReLU(x):
  data = [max(0,value) for value in x]
  return np.array(data, dtype=float)

# Derivative for ReLU
def ReLU_derivative(x):
  data = [1 if value>0 else 0 for value in x]
  return np.array(data, dtype=float)