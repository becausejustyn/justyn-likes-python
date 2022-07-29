
# Logistic Regression from Scratch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Here, we write the code for the aforementioned sigmoid (logit) function. It is important to note that this function can be applied 
# to all of the elements of a `numpy` array individually, simply because we make use of the exponential function from the **NumPy** package.

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost

'''
Next, we write the cost function for logistic regression. Note that the cost function used in logistic regression is different than the one used in linear regression. 

Remember, in linear regression we calculated the weighted sum of input data and parameters and fed that sum to the cost function to calculate the cost. 
When we plotted the cost function it was seen to be convex, hence a local minimum was also the global minimum.

However, in logistic regression, we apply sigmoid function to the weighted sum which makes the resulting outcome non-linear. 
If we feed that non-linear outcome to the cost function, what we get would be a non-convex function and we wouldn't be assured to find only one local minimum that is also the global minimum. 

As a result, we use another cost function to calculate the cost which is guaranteed to give one local minimum during the optimization.
'''

def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y)) 
        cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)

# Gradient descent implementation here is not so different than the one we used in linear regression. Only difference to be noted is the sigmoid function that is applied to the weighted sum.

def predict(X, params):
    return np.round(sigmoid(X @ params))

# While writing out the prediction function, let's not forget that we are dealing with probabilities here. 
# Hence, if the resulting value is above 0.50, we round it up to 1, meaning the data point belongs to the class 1. 
# Consequently, if the probability of a data point belonging to the class 1 is below 0.50, it simply means that it is part of the other class (class 0). 
# Remember that this is binary classification, so we have only two classes (class 1 and class 0).

df = pd.read_csv("~/Downloads/winequality-white.csv", sep=";")
df.columns = df.columns.str.lower().str.replace(' ','_')
df_binary1 = df.copy()
bi_bins = [3,6,10]
labels_name = ['low', 'high']
d = dict(enumerate(labels_name, 1))
df_binary1['quality'] = np.vectorize(d.get)(np.digitize(df_binary1['quality'], bi_bins))

X = df_binary1.drop(['quality'], axis=1).to_numpy()

drop(['B', 'C'], axis=1)

y = df_binary1['quality'].to_numpy()

sns.set_style('white')
sns.scatterplot(X[:,0],X[:,1],hue=y.reshape(-1))

# After coding out the necessary functions, let's create our own dataset with `make_classification` 
# function from `sklearn.datasets`. We create 500 sample points with two classes and plot the dataset with the help of `seaborn` library.
m = len(y)

X = np.hstack((np.ones((m,1)),X))
n = np.size(X,1)
params = np.zeros((n,1))

iterations = 1500
learning_rate = 0.03

initial_cost = compute_cost(X, y, params)

print("Initial Cost is: {} \n".format(initial_cost))

(cost_history, params_optimal) = gradient_descent(X, y, params, learning_rate, iterations)

print("Optimal Parameters are: \n", params_optimal, "\n")

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

# Now, let's run our algorithm and calculate the parameters of our model. Seeing plot, we can now be sure that we have implemented the logistic regression algorithm without a fault, 
# since it decreases with every iteration until the decrease is so minimal that the cost converges to a minimum which is what we want indeed.
y_pred = predict(X, params_optimal)
score = float(sum(y_pred == y))/ float(len(y))

print(score)

# After running the algorithm and getting the optimal parameters, we want to know how successful our model 
# is at predicting the classes of our data. For this reason, we use `accuracy_score` function from `sklearn.metrics` to calculate the accuracy.
slope = -(params_optimal[1] / params_optimal[2])
intercept = -(params_optimal[0] / params_optimal[2])

sns.set_style('white')
sns.scatterplot(X[:,1],X[:,2],hue=y.reshape(-1))

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k")