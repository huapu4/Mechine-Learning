import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf


def validation_curve(X, y, Xval, yval):
    # Selected values of lambda (don't change this)
    lambda_vec = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # You need to return these variables correctly.
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

    for i in range(lambda_vec.size):
        lmd = lambda_vec[i]
        theta = tlr.train_linear_reg(X, y, lmd)
        error_train[i] = lrcf.linear_reg_cost_function(theta, X, y, 0)[0]
        error_val[i] = lrcf.linear_reg_cost_function(theta, Xval, yval, 0)[0]
   
    return lambda_vec, error_train, error_val
