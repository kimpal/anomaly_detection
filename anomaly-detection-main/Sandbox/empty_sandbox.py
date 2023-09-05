import numpy as np
# Hubser Loss Function
def huber_loss(y_pred, y, delta=1.0, n=data_points):
    mse_huber = n*(y - y_pred)**2
    mae_huber = delta * (np.abs(y - y_pred) - n * delta)
    return np.where(np.abs(y - y_pred) <= delta, mse_huber, mae_huber)