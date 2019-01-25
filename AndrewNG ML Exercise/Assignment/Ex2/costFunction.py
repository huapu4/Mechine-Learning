import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size
    cost = 0
    grad = np.zeros(theta.shape)
    
    h = sigmoid(X.dot(theta))  ##hypothesis
    ##  the cost function
    first = -y * np.log(h)     
    second = (1-y) * np.log(1-h)
    cost = np.sum(first - second) / m

    grad = X.T.dot(h - y) / m
    
    return cost, grad
