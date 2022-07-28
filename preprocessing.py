
import numpy as np

def normalize(X, axis=-1, order=2):
    '''Normalize the dataset X'''
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def standardize(X):
    '''Standardize the dataset X'''
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std

def shuffle_data(X, y, seed=None):
    '''Random shuffle of the samples in X and y'''
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


######################    ML Helpers    #########################################

def add_bias(input_matrix):
    n = len(input_matrix)
    bias = np.ones(n, 1)
    return np.concatenate([input_matrix, bias], axis = -1)
        
def add_bias2(data):
    return np.hstack((np.ones((data.shape[0], 1)), data))


def batch_iterator(X, y=None, batch_size=64):
    '''Simple batch generator'''
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    '''Split the data into train and test sets'''
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test