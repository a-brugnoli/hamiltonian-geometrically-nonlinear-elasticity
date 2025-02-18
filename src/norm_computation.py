import numpy as np
import firedrake as fdrk

def error_norm(exact_vec, numerical_vec, time_step, norm="Linf"):

    difference_vec = np.abs(numerical_vec - exact_vec)
    if norm=="Linf":
        return np.max(difference_vec)/np.max(exact_vec)
    elif norm=="L2":
        return np.sqrt(np.sum(time_step*difference_vec**2))
    elif norm=="final":
        return difference_vec[-1]/np.abs(exact_vec[-1]) 
    else:
        raise ValueError("Unknown norm")

def firedrake_error_norm(exact_list_functions, numerical_list_functions, time_step, norm="Linf"):
    n_functions = len(numerical_list_functions)
    assert n_functions==len(exact_list_functions)

    difference_vec = np.array([fdrk.errornorm(exact_list_functions[ii], \
                                               numerical_list_functions[ii]) for ii in range(n_functions)])
    
    exact_vec = np.array([fdrk.norm(exact_list_functions[ii]) for ii in range(n_functions)])

    if norm=="Linf":
        return np.max(difference_vec)/np.max(exact_vec)
    elif norm=="L2":
        return np.sqrt(np.sum(time_step*difference_vec**2))
    elif norm=="final":
        return difference_vec[-1]/np.abs(exact_vec[-1]) 
    else:
        raise ValueError("Unknown norm")