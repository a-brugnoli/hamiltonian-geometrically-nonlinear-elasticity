import numpy as np
from math import pi
import time
import os
from duffing_oscillator import DuffingOscillator
from plot_duffing import plot_results
import matplotlib.pyplot as plt
from src.postprocessing.plot_convergence import plot_convergence
from src.postprocessing.options import configure_matplotib
configure_matplotib()


def error_norm(numerical_vec, exact_vec, time_step, norm="Linf"):

    difference_vec = np.abs(numerical_vec - exact_vec)
    if norm=="Linf":
        return np.max(difference_vec)/np.max(exact_vec)
    elif norm=="L2":
        return np.sqrt(np.sum(time_step*difference_vec**2))
    elif norm=="final":
        return difference_vec[-1]/exact_vec[-1]
    else:
        raise ValueError("Unknown norm")

# Initial condition
q0 = 10

# Pyisical parameters
alpha = 10
beta = 5
omega_0 = np.sqrt(alpha + beta * q0**2)

T = 2*pi/omega_0
t_end = 100*T
# Time parameters
t_span = [0, t_end]

norm_type = "Linf" 
dt_base = T/100
# sec_factor = 1/10
# dt_base = sec_factor*2/omega_0

n_case = 5
log_base = 2
time_step_vec = [dt_base/log_base**n for n in range(n_case)]

output_frequency_vec = [int(dt_base/dt) for dt in time_step_vec]

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
    error_q_leapfrog = error_norm(q_leapfrog, q_exact, time_step=dt, norm=norm_type)
    error_q_dis_gradient = error_norm(q_dis_gradient, q_exact, time_step=dt, norm=norm_type)
    error_q_lin_implicit = error_norm(q_lin_implicit, q_exact, time_step=dt, norm=norm_type)

    error_v_leapfrog = error_norm(v_leapfrog, v_exact, time_step=dt, norm=norm_type)
    error_v_dis_gradient = error_norm(v_dis_gradient, v_exact, time_step=dt, norm=norm_type)
    error_v_lin_implicit = error_norm(v_lin_implicit, v_exact, time_step=dt, norm=norm_type)

    error_E_leapfrog = error_norm(E_leapfrog, E_exact, time_step=dt, norm=norm_type)
    error_E_dis_gradient = error_norm(E_dis_gradient, E_exact, time_step=dt, norm=norm_type)
    error_E_lin_implicit = error_norm(E_lin_implicit, E_exact, time_step=dt, norm=norm_type)

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

    # dict_position = {"Exact": q_exact, "Leapfrog": q_leapfrog,\
    #                 "Discrete gradient": q_dis_gradient, "Linear implicit": q_lin_implicit}
    # dict_velocity = {"Exact": v_exact, "Leapfrog": v_leapfrog, \
    #                 "Discrete gradient": v_dis_gradient, "Linear implicit": v_lin_implicit}
    # dict_energy = {"Exact": E_exact, "Leapfrog": E_leapfrog, \
    #                 "Discrete gradient": E_dis_gradient, "Linear implicit": E_lin_implicit}
    # dict_results = {"time": t_vec, "position": dict_position, "velocity": dict_velocity, "energy": dict_energy}

    # plot_results(dict_results, explicit=True)


directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)

dict_position = {"Leapfrog": error_vec_q_leapfrog,\
                "Discrete gradient": error_vec_q_dis_gradient,\
                "Linear implicit": error_vec_q_lin_implicit}
    
dict_velocity = {"Leapfrog": error_vec_v_leapfrog,\
                "Discrete gradient": error_vec_v_dis_gradient,\
                "Linear implicit": error_vec_v_lin_implicit}


str_xlabel = '$\log \Delta t \; \mathrm{[s]}$'
plot_convergence(time_step_vec, dict_position, xlabel=str_xlabel, ylabel="$\log \Delta q$", \
                title='Position error', savefig=f"{directory_results}convergence_position.pdf")
plot_convergence(time_step_vec, dict_velocity, xlabel=str_xlabel, ylabel="$\log \Delta v$",  \
                 title='Velocity error', savefig=f"{directory_results}convergence_velocity.pdf")

plt.figure()
plt.loglog(time_step_vec, error_vec_E_leapfrog, '*-', label='Leapfrog')
plt.loglog(time_step_vec, error_vec_E_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, error_vec_E_lin_implicit, '+-', label='Linear implicit')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel("$\log \Delta H$")

plt.legend()
plt.grid(True)
plt.title("Energy error")
plt.savefig(f"{directory_results}energy_error.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.figure()
plt.loglog(time_step_vec, elapsed_vec_leapfrog, '*-', label='Leapfrog')
plt.loglog(time_step_vec, elapsed_vec_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, elapsed_vec_lin_implicit, '+-', label='Linear implicit')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel("$\log \\tau$")
plt.legend()
plt.grid(True)
plt.title("Computational time [ms]")
plt.savefig(f"{directory_results}computational_time.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.show()
