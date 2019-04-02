import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    # Useful values
    m = x.shape[0]
    num_labels = theta2.shape[0]

    # You need to return the following variable correctly
    p = np.zeros(m)

    x = np.c_[np.ones(m), x]    #增加theta_0
    #hypothesis 1、2
    h1 = sigmoid(x.dot(theta1.T))   
    h1 = np.c_[np.ones(h1.shape[0]), h1]
    h2 = sigmoid(np.dot(h1, theta2.T))
    p = np.argmax(h2, axis=1) + 1


    return p


