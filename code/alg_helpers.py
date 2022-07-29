
from typing import List
#import numpy as np
from math import exp, log, sqrt
from copy import deepcopy
from statistics import mean

#########################  Logistic Regression  ##########################################

# Function which turns vectors of `beta` and `x` values into a value between 0 and 1
def squish(beta: List[float], x: List[float]) -> float:
    assert len(beta) == len(x)
    # Calculate the dot product
    dot_result: float = beta @ x #dot(beta, x)
    # Use sigmoid to get a result between 0 and 1
    return sigmoid(dot_result)

assert squish([1, 2, 3, 4], [5, 6, 7, 8]) == 1.0

# Rescales the data so that each item has a mean of 0 and a standard deviation of 1
# See: https://en.wikipedia.org/wiki/Standard_score
def z_score(data: List[List[float]]) -> List[List[float]]:
    def mean(data: List[float]) -> float:
        return sum(data) / len(data)

    def standard_deviation_sample(data: List[float]) -> float:
        num_items: int = len(data)
        mu: float = mean(data)
        return sqrt(1 / (num_items - 1) * sum([(item - mu) ** 2 for item in data]))

    data_copy: List[List[float]] = deepcopy(data)
    data_transposed = list(zip(*data_copy))
    mus: List[float] = []
    stds: List[float] = []
    for item in data_transposed:
        mus.append(mean(list(item)))
        stds.append(standard_deviation_sample(list(item)))

    for item in data_copy:
        mu: float = mean(item)
        std: float = standard_deviation_sample(item)
        for i, elem in enumerate(item):
            if stds[i] > 0.0:
                item[i] = (elem - mus[i]) / stds[i]

    return data_copy
# xs = z_score(xs)



def estimate_beta(xs, ys, epochs: int = 5000, learning_rate: float = 0.01):
    # Find the best separation to classify the data points
    beta: List[float] = [random() / 10 for _ in range(3)]
    print(f'Starting with "beta": {beta}')

    for epoch in range(epochs):
        # Calculate the "predictions" (squishified dot product of `beta` and `x`) based on our current `beta` vector
        ys_pred: List[float] = [squish(beta, x) for x in xs]

        # Calculate and print the error
        if epoch % 1000 == True:
            loss: float = error(ys, ys_pred)
            print(f'Epoch {epoch} --> loss: {loss}')

        # Calculate the gradient
        grad: List[float] = [0 for _ in range(len(beta))]
        for x, y in zip(xs, ys):
            err: float = squish(beta, x) - y
            for i, x_i in enumerate(x):
                grad[i] += (err * x_i)
        grad = [1 / len(x) * g_i for g_i in grad]

        # Take a small step in the direction of greatest decrease
        beta = [b + (gb * -learning_rate) for b, gb in zip(beta, grad)]
    print(f'Best estimate for "beta": {beta}')
