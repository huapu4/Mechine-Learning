import numpy as np


def linear_reg_cost_function(theta, x, y, lmd):
    # Initialize some useful values
    m = y.size
    num = m * 2
    # You need to return the following variables correctly
    cost = 0
    grad = np.zeros(theta.shape)
    error = np.dot(x, theta) - y
    error2 = error ** 2
    cost = np.sum(error2) / num + np.sum(theta[1:] ** 2) * lmd / num
    reg_term = theta * lmd / m 
    reg_term[0] = 0
    grad = np.dot(x.T, error) / m +reg_term
    
    
    return cost, grad
