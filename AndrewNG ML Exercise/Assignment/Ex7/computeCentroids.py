import numpy as np


def compute_centroids(X, idx, K):
    # Useful values
    (m, n) = X.shape

    # You need to return the following variable correctly.
    centroids = np.zeros((K, n))
    
    for k in range(K):
        k_x = X[np.where(idx == k)]
        centroid = np.sum(k_x, axis=0) / k_x.shape[0]
        centroids[k] = centroid
    
    return centroids
