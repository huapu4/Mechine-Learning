import numpy as np
from sigmoid import *


def sigmoid_gradient(z):
    g = np.zeros(z.shape)
    g = sigmoid(z) * (1 - sigmoid(z))

    return g
