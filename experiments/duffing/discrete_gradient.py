import numpy as np
from numpy.linalg import norm

def midpoint_discrete_gradient(x_new, x_old):
    """
    Midpoint discrete gradient 
    """
    alpha = 1
    beta = 1
    H = lambda xi : 1/2 * alpha * xi**2 + 1/4 * beta * xi**4
    dH = lambda xi : alpha * xi + beta * xi**3
    
    x_mid = 0.5*(x_old + x_new)
    x_diff = x_new - x_old
    dH_discrete = dH(x_mid) \
         + x_diff * (H(x_new) - H(x_old) - np.dot(dH(x_mid), x_diff))/norm(x_diff)**2

    return dH_discrete

