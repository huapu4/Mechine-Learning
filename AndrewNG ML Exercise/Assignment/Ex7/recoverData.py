import numpy as np


def recover_data(Z, U, K):
    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    ure = U[:,np.arange(K)]
    X_rec = np.dot(Z, ure.T)

    return X_rec
