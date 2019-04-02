import numpy as np
from sigmoid import *


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1) # 25 * 401
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)   #10 * 26

    # Useful value
    m = y.size   #5000

    # You need to return the following variables correctly
    cost = 0
    theta1_grad = np.zeros(theta1.shape)  # 25 x 401
    theta2_grad = np.zeros(theta2.shape)  # 10 x 26

    Y = np.zeros((m, num_labels)) # 5000*10
    for i in range(m):
        Y[i, y[i]-1] = 1
        
    a1 = np.c_[np.ones(m), X]    # X is (5000,400), m is 5000, 5000 * 401
    a2 = np.c_[np.ones(m), sigmoid(np.dot(a1, theta1.T))]  # 5000 * 26
    
    h = sigmoid(np.dot(a2, theta2.T))  #5000 * 10
   
    reg_t1 = theta1[:,1:]  #25*400
    reg_t2 = theta2[:,1:]  #10*25
    
    first = -Y * np.log(h)
    second = (1-Y) * np.log(1-h)
    cost = np.sum(first - second) / m   # cost
    
    cost += (lmd / (2 * m)) * (np.sum(reg_t1 * reg_t1) + np.sum(reg_t2 * reg_t2))  # add reg
    
    
    #backpropagation
    
    error3 = h - Y # 5000*10
    error2 = np.dot(error3, theta2) *  (a2 * (1 - a2))
    error2 = error2[:, 1:]

    delta1 = np.dot(error2.T, a1)
    delta2 = np.dot(error3.T, a2)
    
    p1 = (lmd / m) * np.c_[np.zeros(hidden_layer_size), reg_t1]
    p2 = (lmd / m) * np.c_[np.zeros(num_labels), reg_t2]

    theta1_grad = p1 + (delta1 / m)
    theta2_grad = p2 + (delta2 / m)
    
    # Unroll gradients
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])
    
    
    

    return cost, grad
