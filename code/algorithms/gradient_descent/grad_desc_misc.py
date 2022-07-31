
from random import random
from typing import List

# The function which predicts a `y` value based on `x` and the `alpha` and `beta` parameters
def predict(alpha: float, beta: List[float], x: List[float]) -> float:
    assert len(beta) == len(x)
    # Prepare data so that we can easily do a dot product calculation
    # Prepend `alpha` to the `beta` vector
    beta: List[float] = beta.copy()
    beta.insert(0, alpha)
    # Prepend a constant (1) to the `x` vector
    x: List[float] = x.copy()
    x.insert(0, 1)
    # Calculate the y value via the dot product (https://en.wikipedia.org/wiki/Dot_product)
    return sum([a * b for a, b in zip(x, beta)])

# (5 * 1) + (1 * 3) + (2 * 4) = 16 <-- the 5 and 1 are the prepended `alpha` and constant values
assert predict(5, [1, 2], [3, 4]) == 16

# SSE (sum of squared estimate of errors), the function we use to calculate how "wrong" we are
# "How much do the actual y values (`ys`) differ from our predicted y values (`ys_pred`)?"
def sum_squared_error(ys: List[float], ys_pred: List[float]) -> float:
    assert len(ys) == len(ys_pred)
    return sum([(y - y_p) ** 2 for y, y_p in zip(ys, ys_pred)])

assert sum_squared_error([1, 2, 3], [4, 5, 6]) == 27

def gradient_descent(xs: List[float], ys: List[float], ys_pred: List[float], epochs: int = 1000, learning_rate: float = 0.00001):
    # Find the best fitting hyperplane through the data points via Gradient Descent
    #epochs: int = 1000
    #learning_rate: float = 0.00001
    alpha: float = random()
    beta: List[float] = [random(), random()]
    print(f'Starting with "alpha": {alpha}')
    print(f'Starting with "beta": {beta}')

    for epoch in range(epochs):
    # Calculate predictions for `y` values given the current `alpha` and `beta`
        ys_pred: List[float] = [predict(alpha, beta, x) for x in xs]

    # Calculate and print the error
        if epoch % 100 == True:
            loss = sum_squared_error(ys, ys_pred)
            print(f'Epoch {epoch} --> loss: {loss}')
    
        # Calculate the gradient
        x: List[float]
        y: List[float]
        # Taking the (partial) derivative of SSE with respect to `alpha` results in `2 (y_pred - y)`
        grad_alpha: float = sum([2 * (predict(alpha, beta, x) - y) for x, y in zip(xs, ys)])
        # Taking the (partial) derivative of SSE with respect to `beta` results in `2 * x (y_pred - y)`
        grad_beta: List[float] = list(range(len(beta)))
        for x, y in zip(xs, ys):
            error: float = (predict(alpha, beta, x) - y)
            for i, x in enumerate(x):
                grad_beta[i] = 2 * error * x

        # Take a small step in the direction of greatest decrease
        alpha = alpha + (grad_alpha * -learning_rate)
        beta = [b + (gb * -learning_rate) for b, gb in zip(beta, grad_beta)]

    print(f'Best estimate for "alpha": {alpha}')
    print(f'Best estimate for "beta": {beta}')
