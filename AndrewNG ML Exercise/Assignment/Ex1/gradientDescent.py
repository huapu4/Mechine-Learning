import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):      
        error = (np.dot(X, theta)) - y  #误差
        theta -= (alpha/m)*np.sum(X*error[:, np.newaxis], 0) #误差迭代
        
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
#        
        
        error = (np.dot(X, theta)) - y
        theta -= (alpha/m)*np.sum(X*error[:, np.newaxis], 0)
        
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
    