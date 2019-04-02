import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]

    for i in range(num_labels):
       
        print('Optimizing for handwritten number {}...'.format(i))
        first_theta = np.zeros((n+1, 1))
        if i < 10:
            i_class = i
        else : i_class = 10
        y_i = np.array([1 if x == i_class else 0 for x in y])
        
        def cost_func(theta_t):
            return lCF.lr_cost_function(theta_t, X, y_i, lmd)[0]
        
        def grad_func(theta_t):
            return lCF.lr_cost_function(theta_t, X, y_i, lmd)[1]
        
        theta = opt.fmin_cg(cost_func, fprime=grad_func, 
                                     x0=first_theta, maxiter=100, 
                                     disp=False, full_output=True)
        theta, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=first_theta, maxiter=100, disp=False, full_output=True)
        
        all_theta[i] = theta
        
        print('Done')

    return all_theta
