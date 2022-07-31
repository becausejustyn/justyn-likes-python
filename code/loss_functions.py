
import numpy as np
import math

##########################   Loss Functions   ##################################

def mean_squared_error(y_true, y_pred):
    '''Returns the mean squared error between y_true and y_pred'''
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
def cost(theta, X, y, learningRate, reg: bool = False):
    theta = np.matrix(theta)
    X, y = (np.matrix(X), np.matrix(y))
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    if reg == False:
        return np.sum(first - second) / (len(X))
    else:
        reg_v = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
        np.sum(first - second) / (len(X)) + reg_v

def calculate_entropy(y):
    '''Calculate the entropy of label array y'''
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

# Cross-entropy loss function.
def cross_entropy(A, Y):
    return -np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))

def accuracy_score(y_true, y_pred):
    '''Compare y_true to y_pred and return the accuracy'''
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def accuracy(y_true, y_pred):
    # both are not one hot encoded
    return np.mean(y_pred == y_true)
    #return np.mean(y_pred == y_true)  

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fp)

assert precision(70, 4930, 13930, 981070) == 0.014

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)

assert recall(70, 4930, 13930, 981070) == 0.005

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)


def mean_absolute_percentage_error(y_true, y_pred, sample_weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)

    if np.any(y_true==0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)

    if type(sample_weights) == type(None):
        return(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return(100/sum(sample_weights)*np.dot(
                sample_weights, (np.abs((y_true - y_pred) / y_true))
        ))
