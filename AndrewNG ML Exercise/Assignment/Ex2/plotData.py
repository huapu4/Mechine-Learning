import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()
    positive = np.where(y==1)
    negative = np.where(y==0)
    
    plt.scatter(X[positive, 0], X[positive, 1], c='b', marker='x')
    plt.scatter(X[negative, 0], X[negative, 1], c='y', marker='o')
    
    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #

