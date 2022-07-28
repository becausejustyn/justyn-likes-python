
import numpy as np
import math

# activation functions

#Defining a sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z)) 

def sigmoid2(x):
    """
    evaluate the boltzman function with midpoint xmid and time constant tau
    over x
    """
    return 1. / (1. + np.exp(-x))

def logistic(x: float) -> float:
    return 1.0 / (1 + math.exp(-x))

def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)