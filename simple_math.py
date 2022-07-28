
import numpy as np


def calculate_variance(X):
    '''Return the variance of the features in dataset X'''
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance

def calculate_std_dev(X):
    '''Calculate the standard deviations of the features in dataset X'''
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev
        





