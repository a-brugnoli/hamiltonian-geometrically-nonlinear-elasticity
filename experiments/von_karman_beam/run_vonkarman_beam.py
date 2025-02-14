import firedrake as fdrk
import time
from von_karman_beam import VonKarmanBeam
import numpy as np
import matplotlib.pyplot as plt
from src.postprocessing.animation_1d_line import create_1d_line_animation
from src.postprocessing.plot_convergence import plot_convergence
from src.postprocessing.plot_surface import plot_surface_from_matrix
from math import pi

width = 0.002 # m
height = 0.002 # m
# A = width*height # Cross section
# I = (width * height**3) / 12 # Second moment of inertia
# rho = 2700 # Density [kg/m^3]
# E = 70*10**9 # Young modulus [Pa]

A = 1 # Cross section
I = 1 # Second moment of inertia
rho = 1 # Density [kg/m^3]
E = 1 # Young modulus [Pa]

wave_speed_bending = np.sqrt(E*I/(rho*A))
wave_speed_traction = np.sqrt(E/rho)
# Initial condition
qz_0 = 100*height
# Time parameters
dt_base = 1e-6

n_sim_output = 500
n_case = 1
log_base = 2
time_step_vec = np.array([dt_base/log_base**n for n in range(n_case)])
print(f"dt list: {time_step_vec}")

error_vec_q_leapfrog = np.zeros(n_case)
error_vec_v_leapfrog = np.zeros(n_case)
error_vec_E_leapfrog = np.zeros(n_case)
elapsed_vec_leapfrog = np.zeros(n_case)

for ii in range(n_case):
    dt = time_step_vec[ii]

    n_elements = 20
    mesh_size_min_bending = np.sqrt(2*dt*wave_speed_bending)
    mesh_size_min_traction = dt*wave_speed_traction

    # mesh_size_min = max(mesh_size_min_bending, mesh_size_min_traction)
    mesh_size_min = mesh_size_min_bending

    # In the linear case the minimum security coefficient is 4
    sec_coeff = 4
    mesh_size = sec_coeff*mesh_size_min
    
    L = mesh_size*n_elements

    omega1_bending = (pi/L)**2*wave_speed_bending
    T1_bending = 2*pi/omega1_bending

    t_end = T1_bending
    t_span = [0, t_end]

    beam = VonKarmanBeam(time_step=dt, n_elem = n_elements, q0=qz_0, \
                        rho = rho, E = E, I = I, A=A, L=L, t_span=t_span, \
                        n_output= n_sim_output)
    
    t_vec_output = beam.t_vec_output*1e3

    # The coefficient depends on the magnitude of the time step (in this case milliseconds)
    interval = 1e3 * beam.output_frequency * dt

    t0_implicit = time.perf_counter()
    dict_results = beam.implicit_method(save_vars=True, type="implicit midpoint")
    tf_implicit = time.perf_counter()

    # t0_leapfrog = time.perf_counter()
    # dict_results = beam.leapfrog(save_vars=True)
    # tf_leapfrog = time.perf_counter()

    energy_vec = dict_results["energy"]

    plt.figure()
    plt.plot(t_vec_output, energy_vec)
    # plt.plot(t_vec_output[1:], energy_vec)
    plt.xlabel("Time [ms]")
    plt.title("Energy leapfrog")

    q_z_list = dict_results["ver_displacement"]
    q_z_array = beam.convert_functions_to_array(q_z_list)

    # anim_z = create_1d_line_animation(t_vec_output_ms, beam.x_vec, q_z_array, interval=interval)

    fig_z, ax_z = plot_surface_from_matrix(t_vec_output, beam.x_vec, q_z_array, \
                                            x_label="$x \; \mathrm{[m]}$", \
                                            y_label="$t \; \mathrm{[ms]}$", \
                                            z_label="$q_z \; \mathrm{[m]}$ ", \
                                            title="Vertical displacement")
    
    q_x_list = dict_results["hor_displacement"]
    q_x_array = beam.convert_functions_to_array(q_x_list)

    # anim_x = create_1d_line_animation(t_vec_output_ms, beam.x_vec, q_x_array, interval=interval)

    fig_z, ax_z = plot_surface_from_matrix(t_vec_output, beam.x_vec, q_x_array, \
                                                x_label="$x$", y_label="$t$", z_label="$q_x$", \
                                                title="Horizontal displacement")
    
    plt.show()