import numpy as np
import time
from duffing_oscillator import DuffingOscillator
from src.postprocessing.options import configure_matplotib
from src.norm_computation import error_norm_time
configure_matplotib()
import pickle
from experiments.duffing.parameters_duffing import *

error_vec_q_leapfrog = np.zeros(n_case)
error_vec_v_leapfrog = np.zeros(n_case)
error_vec_E_leapfrog = np.zeros(n_case)
elapsed_vec_leapfrog = np.zeros(n_case)

error_vec_q_dis_gradient = np.zeros(n_case)
error_vec_v_dis_gradient = np.zeros(n_case)
error_vec_E_dis_gradient = np.zeros(n_case)
elapsed_vec_dis_gradient = np.zeros(n_case)

error_vec_q_lin_implicit = np.zeros(n_case)
error_vec_v_lin_implicit = np.zeros(n_case)
error_vec_E_lin_implicit = np.zeros(n_case)
elapsed_vec_lin_implicit = np.zeros(n_case)

for ii in range(n_case):

    print(f"Case {ii+1}")
    dt = time_step_vec[ii]
    # output_frequency = output_frequency_vec[ii]
    output_frequency = 1

    duffing = DuffingOscillator(alpha, beta, t_span, dt, q0)

    t_vec_all = duffing.t_vec
    t_vec = t_vec_all[::output_frequency]

    # Compute exact solution and numerical solutions
    q_exact_all, v_exact_all = duffing.exact_solution()
    q_exact = q_exact_all[::output_frequency]
    v_exact = v_exact_all[::output_frequency]
    E_exact = duffing.hamiltonian(q_exact, v_exact)

    t0_leapfrog = time.perf_counter()
    q_leapfrog_all, v_leapfrog_all = duffing.leapfrog()
    tf_leapfrog = time.perf_counter()
    q_leapfrog = q_leapfrog_all[::output_frequency]
    v_leapfrog = v_leapfrog_all[::output_frequency]

    t0_dis_gradient = time.perf_counter()
    q_dis_gradient_all, v_dis_gradient_all = duffing.implicit_method("midpoint discrete gradient")
    tf_dis_gradient = time.perf_counter()
    q_dis_gradient = q_dis_gradient_all[::output_frequency]
    v_dis_gradient = v_dis_gradient_all[::output_frequency]

    t0_lin_implicit = time.perf_counter()
    q_lin_implicit_all, x_lin_implicit_all = duffing.linear_implicit_static_condensation()
    tf_lin_implicit = time.perf_counter()
    q_lin_implicit = q_lin_implicit_all[::output_frequency]
    x_lin_implicit = x_lin_implicit_all[::output_frequency, :]
    v_lin_implicit = x_lin_implicit[:, 0]

    elapsed_leapfrog = (tf_leapfrog - t0_leapfrog)*1e3
    elapsed_dis_gradient = (tf_dis_gradient - t0_dis_gradient)*1e3
    elapsed_lin_implicit = (tf_lin_implicit - t0_lin_implicit)*1e3

    print(f"Elapsed time Leapfrog [ms]: {elapsed_leapfrog}")
    print(f"Elapsed time Midpoint Discrete gradient [ms]: {elapsed_dis_gradient}")
    print(f"Elapsed time Linear implicit [ms]: {elapsed_lin_implicit}")

    # Compute energies
    E_leapfrog = duffing.hamiltonian(q_leapfrog, v_leapfrog)
    E_dis_gradient = duffing.hamiltonian(q_dis_gradient, v_dis_gradient)
    E_lin_implicit =  np.einsum('ij,ij->i', 0.5*x_lin_implicit @ duffing.energy_matrix(), x_lin_implicit)

    # Compute error
    error_q_leapfrog = error_norm_time(q_exact, q_leapfrog, time_step=dt, norm=norm_type)
    error_q_dis_gradient = error_norm_time(q_exact, q_dis_gradient, time_step=dt, norm=norm_type)
    error_q_lin_implicit = error_norm_time(q_exact, q_lin_implicit, time_step=dt, norm=norm_type)

    error_v_leapfrog = error_norm_time(v_exact, v_leapfrog, time_step=dt, norm=norm_type)
    error_v_dis_gradient = error_norm_time(v_exact, v_dis_gradient, time_step=dt, norm=norm_type)
    error_v_lin_implicit = error_norm_time(v_exact, v_lin_implicit, time_step=dt, norm=norm_type)

    error_E_leapfrog = error_norm_time(E_exact, E_leapfrog, time_step=dt, norm=norm_type)
    error_E_dis_gradient = error_norm_time(E_exact, E_dis_gradient, time_step=dt, norm=norm_type)
    error_E_lin_implicit = error_norm_time(E_exact, E_lin_implicit, time_step=dt, norm=norm_type)

    error_vec_q_leapfrog[ii] = error_q_leapfrog
    error_vec_v_leapfrog[ii] = error_v_leapfrog
    error_vec_E_leapfrog[ii] = error_E_leapfrog
    elapsed_vec_leapfrog[ii] = elapsed_leapfrog

    error_vec_q_dis_gradient[ii] = error_q_dis_gradient
    error_vec_v_dis_gradient[ii] = error_v_dis_gradient
    error_vec_E_dis_gradient[ii] = error_E_dis_gradient
    elapsed_vec_dis_gradient[ii] = elapsed_dis_gradient

    error_vec_q_lin_implicit[ii] = error_q_lin_implicit
    error_vec_v_lin_implicit[ii] = error_v_lin_implicit
    error_vec_E_lin_implicit[ii] = error_E_lin_implicit
    elapsed_vec_lin_implicit[ii] = elapsed_lin_implicit


dict_time = {"Time":t_vec}
with open(file_time, "wb") as f:
        pickle.dump(dict_time, f)

dict_position = {"Exact": q_exact, \
                "Discrete gradient": q_dis_gradient,\
                "Linear implicit": q_lin_implicit, \
                "Leapfrog": q_leapfrog}

with open(file_results_position, "wb") as f:
        pickle.dump(dict_position, f)

dict_velocity = {"Exact": v_exact, \
                "Discrete gradient": v_dis_gradient,\
                "Linear implicit": v_lin_implicit, \
                "Leapfrog": v_leapfrog}

with open(file_results_velocity, "wb") as f:
        pickle.dump(dict_velocity, f)

dict_energy = {"Exact": E_exact, \
                "Discrete gradient": E_dis_gradient,\
                "Linear implicit": E_lin_implicit, \
                "Leapfrog": E_leapfrog}

with open(file_results_energy, "wb") as f:
        pickle.dump(dict_energy, f)

dict_error_position = {"Discrete gradient": error_vec_q_dis_gradient,\
                "Linear implicit": error_vec_q_lin_implicit, \
                "Leapfrog": error_vec_q_leapfrog}

with open(file_results_error_position, "wb") as f:
        pickle.dump(dict_error_position, f)


dict_error_velocity = {"Discrete gradient": error_vec_v_dis_gradient,\
                "Linear implicit": error_vec_v_lin_implicit, \
                "Leapfrog": error_vec_v_leapfrog}

with open(file_results_error_velocity, "wb") as f:
        pickle.dump(dict_error_velocity, f)

dict_error_energy = {"Discrete gradient": error_vec_E_dis_gradient,\
                "Linear implicit": error_vec_E_lin_implicit, \
                "Leapfrog": error_vec_E_leapfrog}

with open(file_results_error_energy, "wb") as f:
        pickle.dump(dict_error_energy, f)

dict_elapsed_time = {"Discrete gradient": elapsed_vec_dis_gradient,\
                "Linear implicit": elapsed_vec_lin_implicit,\
                "Leapfrog": elapsed_vec_leapfrog}

with open(file_results_comp_time, "wb") as f:
        pickle.dump(dict_elapsed_time, f)



