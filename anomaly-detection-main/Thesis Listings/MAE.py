import numpy as np

# MAE loss function
def loss_MAE(y_pred, y_true):
    absolute_error = np.abs(y_pred - y_true)
    sum_absolute_error = np.sum(absolute_error)
    loss = sum_absolute_error / y_true.size
    return loss