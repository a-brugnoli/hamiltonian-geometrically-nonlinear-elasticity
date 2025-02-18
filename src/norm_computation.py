import numpy as np
import firedrake as fdrk

def error_norm_time(exact_vec, numerical_vec, time_step, norm="Linf"):

    difference_vec = np.abs(numerical_vec - exact_vec)
    if norm=="Linf":
        return np.max(difference_vec)/np.max(exact_vec)
    elif norm=="L2":
        return np.sqrt(np.sum(time_step*difference_vec**2))
    elif norm=="final":
        return difference_vec[-1]/np.abs(exact_vec[-1]) 
    else:
        raise ValueError("Unknown norm")
    


def error_norm_space_time(exact_array, numerical_array, time_step, norm="Linf"):

    difference_array = np.abs(numerical_array - exact_array)
    norm_space_difference = np.linalg.norm(difference_array, axis=1)
    norm_space_exact = np.linalg.norm(exact_array, axis=1)

    if norm=="Linf":
        return np.max(norm_space_difference)/np.max(norm_space_exact)
    elif norm=="L2":
        return np.sqrt(np.sum(time_step*norm_space_difference**2))
    elif norm=="final":
        return norm_space_difference[-1]/np.abs(norm_space_exact[-1]) 
    else:
        raise ValueError("Unknown norm")