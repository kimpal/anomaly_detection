import numpy as np

# Loss Function - Mean Squared Error
def loss_mse(y_pred, y_true):
    squared_err = (y_pred - y_true) ** 2
    sum_squared_err = np.sum(squared_err)
    loss = sum_squared_err / y_true.size
    return loss