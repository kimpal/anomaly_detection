import numpy as np

# Hyperbolic Tangent (tanh) Activation Function
def tanh(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# htan derivative
def tanh_derivative(x):
  return 1 - tanh(x) * tanh(x)