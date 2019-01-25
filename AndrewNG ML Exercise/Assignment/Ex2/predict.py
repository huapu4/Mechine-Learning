import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]

    p = np.zeros(m)

    p = sigmoid(X.dot(theta))
    positive = np.where(p >= 0.5)
    negative = np.where(p < 0.5)
    
    p[positive] = 1
    p[negative] = 0

    return p
