# Calculates the Cross Entropy
def Cross_Entropy(y_hat, y):
    if y == 1:
        return -log(y_hat)
    else:
        return -log(1 - y_hat)