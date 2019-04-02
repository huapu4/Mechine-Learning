import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    h = sigmoid(X.dot(theta))
    reg_theta = theta[1:]
    first = np.sum(-y * np.log(h))
    second = np.sum((1-y) * np.log(1-h))
    reg = lmd * np.sum(reg_theta * reg_theta) / (2 * m)

    cost =  (first - second) / m + reg
    
    grad_1d = X.T.dot(h - y) / m

    grad[0] = grad_1d[0]
    grad[1:] = grad_1d[1:]  + lmd * reg_theta / m
    
    return cost, grad
