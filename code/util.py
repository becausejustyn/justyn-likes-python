
import numpy as np
from scipy.stats import norm
import math
import pandas as pd

def format_decimals_factory(num_decimals = 1):
    return lambda x: "{1:.{0}f}".format(num_decimals, x)

def map_df(min: int, max: int, variable):
  data = pd.concat([nfl.load_pbp_data(variable).assign(variable = variable) for variable in range(min, max)])
  
def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)
    
# seq function
# print 1, 2, 3
#[_ for _ in range(1, 4, 1)]

def seq(min: int, max: int, n: int):
    '''
    like seq() in R
    this would print 1, 2, 3
    [_ for _ in range(1, 4, 1)]
    using this function seq(1, 4, 1)
    '''
    return [_ for _ in range(min, max+1, n)]

def dnorm(x, mean=0, sd=1, log=False):
    """
    Density of the normal distribution with mean 
    equal to mean and standard deviation equation to sd
    same functions as rnorm in r: ``dnorm(x, mean=0, sd=1, log=FALSE)``
    :param x: the vector od quantiles
    :param mean: vector of means
    :param sd: vector of standard deviations
    :return: the list of the density  
    :author: Wenqiang Feng
    :email:  von198@gmail.com    
    """
    if log:
        return np.log(norm.pdf(x=x, loc=mean, scale=sd))
    else:
        return norm.pdf(x=x, loc=mean, scale=sd)

def prob_dens(a, m, sd):
  return dnorm(a, m, sd)

def prob(vec, pm, psd):
  return math.prod(map(prob_dens, (vec, pm, psd)))

### An example of different function options instead of doing nested ifelse

# The underscores indicate that these are "private" functions, as they're
# intended to be called by our median function but not by other people
# using our statistics library.
def _median_odd(xs: List[float]) -> float:
    """If len(xs) is odd, the median is the middle element"""
    return sorted(xs)[len(xs) // 2]

def _median_even(xs: List[float]) -> float:
    """If len(xs) is even, it's the average of the middle two elements"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2  # e.g. length 4 => hi_midpoint 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
    """Finds the 'middle-most' value of v"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)



from typing import List

Vector = List[float]

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6

def _negative_log_likelihood(x: Vector, y: float, beta: Vector) -> float:
    """The negative log likelihood for one data point"""
    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else:
        return -math.log(1 - logistic(dot(x, beta)))

# one way to map it

def negative_log_likelihood(xs: List[Vector],
                            ys: List[float],
                            beta: Vector) -> float:
    return sum(_negative_log_likelihood(x, y, beta)
               for x, y in zip(xs, ys))
