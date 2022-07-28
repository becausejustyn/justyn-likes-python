
# data wrangling

import numpy as np


# Reshape the data so that we have a one-hot matrix for the targets.
# https://en.wikipedia.org/wiki/One-hot
def one_hot(t):
    one_hot = np.zeros((len(t), 10))
    one_hot[np.arange(len(t)), t] = 1
    return one_hot

def one_hot_encode(labels):
    result = []
    for label in labels:
        if label:
            result.append([0, 1])
        else:
            result.append([1, 0])
    return np.array(result)
 
def to_nominal(x):
    '''
    Conversion from one-hot encoding to nominal
    '''
    return np.argmax(x, axis=1)
    
def to_categorical(x, n_col=None):
    '''One-hot encoding of nominal values'''
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot