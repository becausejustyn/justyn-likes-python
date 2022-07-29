
import numpy as np
import math

# activation functions


# The sigmoid function which turns any number `x` into a number between 0 and 1
# np.exp()
def sigmoid(x: float) -> float:
    return 1. / (1. + math.exp(-x))

assert sigmoid(0) == 0.5

'''
xs_sigmoid: List[float] = [x for x in range(-10, 10)]
ys_sigmoid: List[float] = [sigmoid(x) for x in xs_sigmoid]

plt.plot(xs_sigmoid, ys_sigmoid)
'''

def simple_softmax(x):
  exp = math.exp(x)
  return exp / exp.sum()

def softmax(x):
    # for batch computation, e.g. softmax([[10, 10], [1, 4]])
    result = []
    for instance in x:
        exp = math.exp(instance - max(instance))
        #exp = np.exp(instance - np.max(instance))
        result.append(exp / exp.sum())
    return result

def logistic(x: float) -> float:
    return 1.0 / (1 + math.exp(-x))

def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)

def relu(x):
    # ReLU(x) = x if (x >= 0), 0 if (x < 0)
    # ReLU(x) = max(0, x)
    x[x < 0] = 0
    return x

'''
x = np.linspace(-10, 10, 100)
y = relu(np.linspace(-10, 10, 100))
fig = plt.figure()
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title='ReLU')
ax.plot(x, y, color='skyblue')
plt.show()
'''
