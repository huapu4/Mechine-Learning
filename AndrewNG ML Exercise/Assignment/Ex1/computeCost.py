import numpy as np


def compute_cost(X, y, theta):
    m = y.size #数据的个数
    cost = 0    
    cost = np.sum(np.power(((np.dot(X, theta)) - y), 2)) / (2 * m) #代价函数
    return cost
