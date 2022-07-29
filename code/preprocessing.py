
import numpy as np

def normalize(X, axis=-1, order=2):
    '''Normalize the dataset X'''
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def normalize(data, min_val=0, max_val=1):
    """
    Normalizes values in a list of data points to a range, e.g.,
    between 0.0 and 1.0. 
    Returns the original object if value is not a integer or float.
    
    """
    norm_data = []
    data_min = min(data)
    data_max = max(data)
    for x in data:
        numerator = x - data_min
        denominator = data_max - data_min
        x_norm = (max_val-min_val) * numerator/denominator + min_val
        norm_data.append(x_norm)
    return norm_data


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

from random import seed, randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

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

'''
xs = [x for x in range(1000)]  # xs are 1 ... 1000
ys = [2 * x for x in xs]       # each y_i is twice x_i
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

# Check that the proportions are correct
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

# Check that the corresponding data points are paired correctly.
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))
'''
