

import numpy as np
from typing import List
# this might be messy for now


# Stochastic Gradient Descent using the adagrad approach

def sgd(par, X: List[float], y: List[float], stepsize, stepsize_tau = 0, average = False):
    '''
    Estimate a linear regression via stochastic gradient descent using adagrad

    par: parameter estimates
    X: model matrix
    y: target variable
    stepsize: the learning rate
    stepsize_tau: if > 0, a check on the LR at early iterations
    average: an alternative method
    
    Returns a dictionary of the parameter estimates, the estimates for each iteration,
    the loss for each iteration, the root mean square error, and the fitted values
    using the final parameter estimates
    '''

    beta = par
    betamat = np.zeros_like(X)
    # if you want these across iterations
    fits = np.zeros_like(y)              
    loss = np.zeros_like(y)
    s = 0
  
    for i in np.arange(X.shape[0]):
        Xi = X[i]
        yi = y[i]
        
        LP   = np.dot(Xi, beta)                                
        grad = np.dot(Xi.T, LP - yi)                           
        s    = s + grad**2
        beta -= stepsize * grad/(stepsize_tau + np.sqrt(s)) 
        
        if average and i > 1:
            beta -= 1/i * (betamat[i - 1] - beta)
        
        betamat[i] = beta
        fits[i]    = LP
        loss[i]    = (LP - yi)**2
        
    LP = np.dot(X, beta)
    lastloss = np.dot(LP - y, LP - y)
    
    return({
        # final parameter estimates
        'par': beta,   
        # all estimates                              
        'par_chain': betamat,                        
        # observation level loss       
        'loss': loss,                                
        'RMSE': np.sqrt(np.sum(lastloss)/X.shape[0]),
        'fitted': LP
    })

'''
p.random.seed(1234)

n = 10000
x1 = np.random.normal(size = n)
x2 = np.random.normal(size = n)
y = 1 + .5*x1 + .2*x2 + np.random.normal(size = n)

X = np.column_stack([np.ones(n), x1, x2])

init = np.zeros(3)

fit_sgd = sgd(
  init, 
  X, 
  y,
  stepsize = .1,
  stepsize_tau = .5
)

fit_sgd['par'].round(4)
fit_sgd['RMSE'].round(4)
'''
