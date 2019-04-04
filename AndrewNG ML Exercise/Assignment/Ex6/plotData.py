import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    
    plt.scatter(X[pos, 0], X[pos, 1], marker='x')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o')
    

