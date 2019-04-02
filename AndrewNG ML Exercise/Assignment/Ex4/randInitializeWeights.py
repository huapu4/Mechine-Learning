import numpy as np


def rand_initialization(l_in, l_out):
    # You need to return the following variable correctly
    w = np.zeros((l_out, 1 + l_in))

    e = 0.1
    w = np.random.rand(l_out, 1 + l_in) * (2 * e) - e

    return w
