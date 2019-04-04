import numpy as np


def gaussian_kernel(x1, x2, sigma):
    x1 = x1.flatten()
    x2 = x2.flatten()

    sim = 0
    sim = np.exp(-(np.sum((x1-x2)**2) / (2 * sigma**2 )))

    return sim

