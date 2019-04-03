import numpy as np


def poly_features(X, p):
    # You need to return the following variable correctly.
    X_poly = np.zeros((X.size, p))
    Poly = np.arange(1, p+1)
    X_poly = X.reshape((X.size, 1)) ** Poly
    
    return X_poly