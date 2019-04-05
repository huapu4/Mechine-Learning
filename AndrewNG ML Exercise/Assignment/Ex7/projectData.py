import numpy as np


def project_data(X, U, K):
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    ure = U[:,np.arange(K)]
    Z = np.dot(X, ure)

    return Z
