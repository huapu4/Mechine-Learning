import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size
    cost = 0
    grad = np.zeros(theta.shape)
    
    h = sigmoid(X.dot(theta))    ## hypothesis
    reg_theta = theta[1:] #从1开始才需要加入正则项
    ## the cost function with regularization
    first = -y * np.log(h)
    second = (1-y) * np.log(1-h)
    reg = lmd * np.sum(reg_theta * reg_theta) / (2 * m)
    cost = np.sum(first - second) / m + reg
    
    grad_1d = X.T.dot(h - y) / m

    grad[0] = grad_1d[0]
    grad[1:] = grad_1d[1:]  + lmd * reg_theta / m


    return cost, grad
