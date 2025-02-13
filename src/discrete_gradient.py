import numpy as np
from numpy.linalg import norm
import firedrake as fdrk
from scipy.integrate import simpson, quad

def midpoint_discrete_gradient(x_new, x_old, H, grad_H):
     """
     Midpoint discrete gradient 
     """
     x_mid = 0.5*(x_old + x_new)
     x_diff = x_new - x_old
     dH_discrete = grad_H(x_mid) \
          + x_diff * (H(x_new) - H(x_old) - np.dot(grad_H(x_mid), x_diff))/norm(x_diff)**2

     return dH_discrete


def mean_value_discrete_gradient(x_new, x_old, grad_H):
     """
     Mean value discrete gradient.
     For cubic polynomials Simspon rule is exact if the number of samples 
     is odd and the the points are equally spaced.
     """
     # n_points = 3
     # s_vec = np.linspace(0, 1, n_points)
     # q_s_vec = x_old + s_vec * (x_new - x_old)  
     # integrand = grad_H(q_s_vec)
     # integral = simpson(integrand, s_vec)

     integrand = lambda s: grad_H(x_old + s * (x_new - x_old))
     integral = quad(integrand, 0, 1)[0]

     return integral


def discrete_gradient_firedrake(var_new, var_old, test_var, grad_H, H):
    """
    Midpoint implementation of discrete gradient in firedrake
    Mean value discrete gradient not supported on multiple meshes.
    Perhaps the simplest is to create a tensor product mesh to eliminate the error.
    Parameters:
    var_new: variable at next time step
    grad_H : weak variational derivative
    H : Hamiltonian functional
    
    """
    var_mid = 0.5*(var_old + var_new)
    var_diff = var_new - var_old

    dH_xmid = grad_H(test_var, var_mid)

    num_coeff = fdrk.assemble(H(var_new) - H(var_old) - grad_H(var_diff, var_mid))
    den_coeff = fdrk.norm(var_diff)**2
    coeff = num_coeff/den_coeff
    dH_discrete = dH_xmid + fdrk.inner(test_var, coeff*var_diff) * fdrk.dx

    return dH_discrete