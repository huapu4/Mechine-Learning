import numpy as np


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, lmd):
    X = params[0:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    # You need to set the following values correctly.
    cost = 0
    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)
    miners = (np.dot(X, theta.T) - Y) * R
    cost = np.sum(miners**2) / 2 + lmd / 2 *(np.sum(theta**2) + np.sum(X**2))
    
    X_grad = np.dot(miners, theta) + lmd *X
    theta_grad = np.dot(miners.T, X) + lmd * theta
    
    grad = np.concatenate((X_grad.flatten(), theta_grad.flatten()))

    return cost, grad
