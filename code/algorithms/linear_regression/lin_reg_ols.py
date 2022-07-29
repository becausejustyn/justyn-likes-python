
import numpy as np

def lm_ols(par, X, y):
    '''
    An approach via least squares loss function.
    par: parameters to be estimated
    X: predictor matrix with intercept column
    y: response
    '''

    # setup
    # coefficients  
    beta = par.reshape(3, 1)              
    N = X.shape[0]
    p = X.shape[1]
    
    # linear predictor
    LP = X * beta                              
    # identity link in the glm sense
    mu = LP                                    
    
    # squared error loss
     
    return(np.sum(np.square(y - mu)))
