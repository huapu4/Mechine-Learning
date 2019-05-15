import numpy as np


def select_threshold(yval, pval):
    f1 = 0

    # You have to return these values correctly
    best_eps = 0
    best_f1 = 0

    for epsilon in np.linspace(np.min(pval), np.max(pval), num=1001):
        pre = np.less(pval, epsilon)
        tp = np.sum(np.logical_and(pre, yval))
        fn = np.sum(np.logical_and(np.logical_not(pre), yval))
        fp = np.sum(np.logical_and(pre, yval==0))
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        
        
        if f1 > best_f1:
            best_f1 = f1
            best_eps = epsilon

    return best_eps, best_f1