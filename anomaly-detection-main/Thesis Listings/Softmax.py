from math import exp

# Softmax Activation Function
def SoftMax(inputVector):
    # Calculating the exponent for each element in the input vector
    exponents = [exp(j) for j in inputVector]
    
    # Dividing the exponent of valuue by the sum of the exponents.
    # Round to 3 decimals.
    p = [round(exp(i)/sum(exponents), 3) for i in inputVector]
    return p