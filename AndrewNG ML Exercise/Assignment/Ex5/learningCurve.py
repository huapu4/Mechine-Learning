import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def learning_curve(X, y, Xval, yval, lmd):
    # Number of training examples
    m = X.shape[0]

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        X_i = X[:i+1]
        y_i = y[:i+1]
        theta = tlr.train_linear_reg(X_i, y_i, lmd)
        error_train[i] = lrcf.linear_reg_cost_function(theta, X_i, y_i, 0)[0]
        error_val[i] = lrcf.linear_reg_cost_function(theta, Xval, yval, 0)[0]
        
    
    return error_train, error_val
