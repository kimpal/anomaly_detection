import numpy as np

# Scaled Exponential Linear Units
def SELU(x, lambdaa = 1.0507, alpha = 1.6732):
    if x >= 0:
        return lambdaa * x
    else:
        return lambdaa * alpha * (np.exp(x) - 1)