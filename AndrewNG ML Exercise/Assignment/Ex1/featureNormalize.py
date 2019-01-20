import numpy as np


def feature_normalize(X):
    n = X.shape[1]  # the number of features
    X_norm = X
    mu = np.zeros(n)
    sigma = np.zeros(n)
    # 特征缩放
    mu = np.mean(X,0)
    sigma = np.std(X, 0, ddof=1)
    X_norm = (X-mu) / sigma    
    
    
    return X_norm, mu, sigma
