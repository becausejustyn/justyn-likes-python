
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binom

np.random.seed(123)

def logreg_ml(par, X, y):
    '''
    A maximum likelihood approach.
    par: parameters to be estimated
    X: predictor matrix with intercept column
    y: response
    '''

    # coefficients
    beta   = par.reshape(X.shape[1], 1)            
    # N = X.shape[0]
    
    # linear predictor
    LP = np.dot(X, beta)   
    # logit link                           
    mu = 1/(1 + np.exp(-LP))                   
    
    # calculate likelihood
    L = binom.logpmf(y, 1, mu) 
    # alternate log likelihood form
    # L =  np.multiply(y, np.log(mu)) + np.multiply(1 - y, np.log(1 - mu))    
     # optim by default is minimization, and we want to maximize the likelihood 
    L = -np.sum(L)      
    
    return(L)

# n = sample size, k = n of predictors
N, k = 2500, 2 
# reshape
X = np.matrix(np.random.normal(size = N * k)).reshape(N, k) 
eta = -.5 + .2*X[:, 0] + .1*X[:, 1]

y = np.random.binomial(1, p = 1/(1 + np.exp(-eta)), size = (N, 1))
dfXy = pd.DataFrame(np.column_stack([X, y]), columns = ['X1', 'X2', 'y'])

X_mm = np.column_stack([np.repeat(1, N).reshape(N, 1), X])

fit_ml = minimize(
  fun = logreg_ml,
  x0  = [0, 0, 0],
  args    = (X_mm, y),
  method  = 'BFGS', 
  tol     = 1e-12, 
  options = {'maxiter': 500}
)

pars_ml  = fit_ml.x
