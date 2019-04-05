import numpy as np


def find_closest_centroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    m = X.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(m)
    
    means = np.zeros((m, K))
    
    for i in range(m):
        x = X[i]
        c = x - centroids
        
        for k in range(K):
            means[i, k] = np.linalg.norm(c[k])
    
    idx = np.argmin(means, axis=1)

    return idx
