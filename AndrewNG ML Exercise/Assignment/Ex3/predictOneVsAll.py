import numpy as np


def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variable correctly;
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.c_[np.ones(m), X]

    a = np.dot(all_theta, X.T)
    a = np.roll(a, -1, axis=0)  
    a = np.vstack([np.zeros(m), a])
    p = np.argmax(a, axis=0)
    return p
