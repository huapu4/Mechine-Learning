import numpy as np


def kmeans_init_centroids(X, K):
    # You should return this value correctly
    centroids = np.zeros((K, X.shape[1]))

    num = np.random.randint(X.shape[0], size=K)
    centroids = X[num]
    
    
    return centroids
