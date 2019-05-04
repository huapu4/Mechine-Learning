import numpy as np


def estimate_gaussian(X):
    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)

    return mu, sigma2
