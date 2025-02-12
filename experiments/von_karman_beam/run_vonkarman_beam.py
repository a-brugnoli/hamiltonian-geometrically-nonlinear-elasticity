import firedrake as fdrk
import time
from von_karman_beam import VonKarmanBeam
import numpy as np
import matplotlib.pyplot as plt
# from experiments.tools.animation_1d_function import create_1d_function_animation
from experiments.tools.animation_1d_line import create_1d_line_animation

# Initial condition
qz_0 = 0.001

rho = 1 # Density
A = 1 # Cross section
E = 1 # Young modulus
I = 1 # Second moment of inertia
L = 1 # Length

n_elements = 10
mesh_size = L/n_elements

dt_CFL = 0.5*mesh_size**2*np.sqrt(rho*A/(E*I))

print(f"Critical time: {dt_CFL}")
t_end = 0.01
# Time parameters
sec_coeff = 0.05
dt_base = sec_coeff*dt_CFL
t_span = [0, t_end]

n_case = 1
log_base = 2
dt_vec = np.array([dt_base/log_base**n for n in range(n_case)])
print(f"dt list: {dt_vec}")

sampling_frequency_vec = [int(dt_base/dt) for dt in dt_vec]

error_vec_q_leapfrog = np.zeros(n_case)
error_vec_v_leapfrog = np.zeros(n_case)
error_vec_E_leapfrog = np.zeros(n_case)
elapsed_vec_leapfrog = np.zeros(n_case)

for ii in range(n_case):
    dt = dt_vec[ii]
    beam = VonKarmanBeam(time_step=dt, n_elem = n_elements, q0=qz_0, \
                        rho = rho, E = E, I = I, A=A, L=L, 
                        t_span=t_span)

    t0_leapfrog = time.perf_counter()
    dict_results = beam.leapfrog(save_vars=True)
    tf_leapfrog = time.perf_counter()

    t_vec = dict_results["time"]
    energy_vec = dict_results["energy"]

    q_z_list = dict_results["q_z"]

    x_vec, q_z_array = beam.convert_functions_to_array(q_z_list)

    plt.figure()
    plt.plot(t_vec, energy_vec)
    plt.title("Energy leapfrog")

    # energy_vec_leapfrog = dict_results["energy_leapfrog"]
    # plt.figure()
    # plt.plot(t_vec[1:], energy_vec_leapfrog)
    # plt.title("Perturbed Energy leapfrog")

    # anim = create_1d_function_animation(q_z_list, beam.domain)
    anim = create_1d_line_animation(q_z_array, x_vec)

    plt.show()