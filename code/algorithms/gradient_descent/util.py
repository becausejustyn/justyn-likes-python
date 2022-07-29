
import numpy as np
from scipy.special import expit

def generate_LS_data(n, d, seed=None):
    """ samples random LS data"""
    if seed:
        np.random.seed(3312)

    X = np.random.normal(loc=0, scale=1, size=(n, d))
    y = np.random.normal(loc=0, scale=1, size=n)

    return X, y
  
  
def generate_logreg_data(n, d, seed=None):
    """samples random logistic regression data"""
    if seed:
        np.random.seed(3312)

    # X data with intercept
    X = np.random.normal(loc=0, scale=1, size=(n, d-1))
    X = np.array(np.bmat([X, np.ones((n, 1))]))  # add i ntercept to design matrix

    # true beta
    beta_platon = np.random.normal(loc=0, scale=1, size=d)

    # sample data
    prob = expit(np.dot(A, beta_platon))
    unif = np.random.sample(n)
    y = np.array([1 if unif[i] < prob[i] else -1 for i in range(n)])  # class labels

    return X, y, beta_platon
  
  
def logistic_loss(X, y, beta):
    """ sum_i log(1 + exp(y_i *x_i^T beta))"""
    return np.sum(np.log(1 + np.exp(np.multiply(y, np.dot(X, beta)))))


def logistic_loss_grad(X, y, beta):
    # E = expit(-1*np.multiply(y, np.dot(X, beta)))
    # Eb = np.multiply(E, y).T
    # return np.sum(np.dot(np.diag(Eb), X), axis=0)

    # if y is a scalar need a different format
    if not hasattr(y, "__len__"):
        y01 = int(0 < y)
        p = expit(np.dot(X, beta))
        grad = X * (y01 - p)
    else:
        p = expit(np.dot(X, beta))
        y01 = [1 if label > 0 else 0 for label in y]
        grad = np.dot(X.T, y01 - p)

    return grad
  
