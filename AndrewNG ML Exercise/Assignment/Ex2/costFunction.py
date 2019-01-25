import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)
    

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    # ===========================================================
    h = sigmoid(X.dot(theta))
    first = -y * np.log(h)
    second = (1-y) * np.log(1-h)
    cost = np.sum(first - second) / m

    grad = X.T.dot(h - y) / m
    
    return cost, grad
