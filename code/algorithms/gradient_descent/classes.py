
import numpy as np
from scipy.special import expit
from sklearn import linear_model  # ridge regression

class LeastSquares(object):
    """
    Least squares

    min_beta ||X beta - y||_2^2

    X in R^(n x d)
    y in R^n
    beta in R^d
    """

    def __init__(self, X, y):
        self.name = "least squares"

        # data
        self.X = X
        self.y = y

        self.d = X.shape[1]
        self.n = X.shape[0]

        # lipshitz constant
        self.L_F = np.linalg.norm(X)**2
        self.mu_F = 0

    def F(self, beta):
        return .5*sum((np.dot(self.X, beta) - self.y)**2)

    def grad_F(self, beta):
        return np.dot(self.X.T, np.dot(self.X, beta) - self.y)

    def f(self, beta, i):
        return self.n*.5*(np.dot(self.X[i, :], beta) - self.y[i])**2

    def grad_f(self, beta, i):
        return self.n*(np.dot(self.X[i, :], beta) - self.y[i]) * self.X[i, :]

    def get_solution(self):
        """returns the analytic solution to the LS problem"""
        return np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)),
                      self.X.T), self.y)


class LeastSquaresL2(object):
    """
    Least squares with L2 penalizaion

    min_beta ||X beta - y||_2^2 + alpha ||beta||_2^2

    X in R^(n x d)
    y in R^n
    beta in R^d
    alpha >0
    """

    def __init__(self, X, y, alpha):
        self.name = "least squares with L2 penalty"

        # data
        self.X = X
        self.y = y

        self.d = X.shape[1]
        self.n = X.shape[0]

        # lipshitz constant
        self.L_F = np.linalg.norm(X)**2 + alpha
        self.mu_F = alpha

    def F(self, beta):
        return .5*sum((np.dot(self.X, beta) - self.y)**2) + self.alpha*np.linalg.norm(beta)**2

    def grad_F(self, beta):
        return np.dot(self.X.T, np.dot(self.X, beta) - self.y) + self.alpha*beta

    def f(self, beta, i):
        return self.n*.5*(np.dot(self.X[i, :], beta) - self.y[i])**2 + self.alpha*np.linalg.norm(beta)**2

    def grad_f(self, beta, i):
        return self.n*(np.dot(self.X[i, :], beta) - self.y[i]) * self.X[i, :] + self.alpha*beta

    def get_solution(self):
        """solution based on sklean's ridge regression"""
        # check solution using sklean
        sk_ridge_model = linear_model.Ridge(alpha=self.alpha, fit_intercept=False)
        sk_ridge_model.fit(X=A, y=b)
        return sk_ridge_model.coef_
    
        

class LogisticRegression(object):
    """
    Logistic regression

    min_x sum_i log(1 + exp(b_i *a_i^Tx))

    A in R^(n x d) assumes the last column is all ones for the intercet
    b =+/-1 in R^n
    x in R^d
    """

    def __init__(self, X, y):

        # data
        self.X = y
        self.y = y

        self.d = X.shape[1]
        self.n = X.shape[0]

        # lipshitz constant
        self.L_F = .25 * np.linalg.norm(np.dot(X.T, X))

    def F(self, beta):
        return -1*logistic_loss(self.X, self.y, beta)

    def grad_F(self, beta):
        return -1*logistic_loss_grad(self.X, self.y, beta)

    def f(self, beta, i):
        return -1*self.n*logistic_loss(self.X[i, :], self.y[i], beta)

    def grad_f(self, beta, i):
        return -1*self.n*logistic_loss_grad(self.X[i, :], self.y[i], beta)

