
import numpy as np
from scipy.stats import norm

def lm_ml(par, X, y):
    '''
    A maximum likelihood approach.
    par: parameters to be estimated
    X: predictor matrix with intercept column
    y: response
    '''

    # coefficients
    beta   = par[1:].reshape(X.shape[1], 1)    
    # error sd
    sigma  = par[0]                            
    # N = X.shape[0]
  
    # linear predictor
    LP = X * beta                              
    # identity link in the glm sense
    mu = LP                                    
    
    # calculate likelihood
    # log likelihood; or use norm.logpdf
    L = norm.logpdf(y, loc = mu, scale = sigma) 
    
    # alternate log likelihood form
    #L =  -.5*N*log(sigma2) - .5*(1/sigma2)*crossprod(y-mu)    

     # optim by default is minimization, and we want to maximize the likelihood
    L = -np.sum(L)       
    
    return(L)
