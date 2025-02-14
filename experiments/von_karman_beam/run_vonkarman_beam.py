import firedrake as fdrk
import time
from von_karman_beam import VonKarmanBeam
import numpy as np
import matplotlib.pyplot as plt
from src.postprocessing.animation_1d_line import create_1d_line_animation
from src.postprocessing.plot_convergence import plot_convergence
from src.postprocessing.plot_surface import plot_surface_from_matrix
from math import pi
import os

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)

width = 0.002 # m
height = 0.002 # m
A = width*height # Cross section
I = (width * height**3) / 12 # Second moment of inertia
rho = 2700 # Density [kg/m^3]
E = 70*10**9 # Young modulus [Pa]

# A = 1 # Cross section
# I = 1 # Second moment of inertia
# rho = 1 # Density [kg/m^3]
# E = 1 # Young modulus [Pa]

wave_speed_bending = np.sqrt(E*I/(rho*A))
wave_speed_traction = np.sqrt(E/rho)

L = 1
n_elements = 50
mesh_size = L/n_elements


# # In finite elements, the minimum security coefficient is:
# - 4 for bending 
# - 2 for traction

sec_coeff_bend = 4
sec_coeff_traction = 2

# # If time step is set first, there is a minimum mesh size to guarantee stability
# mesh_size_min_bending = np.sqrt(2*dt*wave_speed_bending)
# mesh_size_min_traction = dt*wave_speed_traction
# mesh_size_min = max(mesh_size_min_bending, mesh_size_min_traction)

# mesh_size = sec_coeff*mesh_size_min
# Given a number of elements the length is determined a posteriori
# L = mesh_size*n_elements

# Here we set the mesh size and the time step follows


dt_max_bending = mesh_size**2/(2*wave_speed_bending)
dt_max_traction = mesh_size/wave_speed_traction

dt_CFL_conservative = min(dt_max_bending/sec_coeff_bend, dt_max_traction/sec_coeff_traction)


# Initial condition
alpha = 1
ampl_hor_disp_0 = alpha*height
ampl_ver_disp_0 = alpha*height
# Time step with security coefficient 4
dt_base = 5*dt_CFL_conservative
omega1_bending = (pi/L)**2*wave_speed_bending
T1_bending = 2*pi/omega1_bending

# dt_base = T1_bending/10000
t_end = T1_bending/5
t_span = [0, t_end]

n_sim_output = 1000
n_case = 1
log_base = 2
time_step_vec = np.array([dt_base/log_base**n for n in range(n_case)])
print(f"dt list: {time_step_vec}")

# error_vec_q_leapfrog = np.zeros(n_case)
# error_vec_v_leapfrog = np.zeros(n_case)
# error_vec_E_leapfrog = np.zeros(n_case)
# elapsed_vec_leapfrog = np.zeros(n_case)

# error_vec_q_dis_gradient = np.zeros(n_case)
# error_vec_v_dis_gradient = np.zeros(n_case)
# error_vec_E_dis_gradient = np.zeros(n_case)
# elapsed_vec_dis_gradient = np.zeros(n_case)

# error_vec_q_implicit_midpoint = np.zeros(n_case)
# error_vec_v_implicit_midpoint = np.zeros(n_case)
# error_vec_E_implicit_midpoint = np.zeros(n_case)
# elapsed_vec_implicit_midpoint = np.zeros(n_case)

for ii in range(n_case):
    dt = time_step_vec[ii]

    
    beam = VonKarmanBeam(time_step=dt, t_span=t_span, n_output= n_sim_output,\
                        n_elem = n_elements, q0_hor = ampl_hor_disp_0, q0_ver=ampl_ver_disp_0, \
                        rho = rho, E = E, I = I, A=A, L=L)
    x_vec = beam.x_vec
    x_point = L/2

    index_point = np.argmin(np.abs(x_vec - x_point))

    
    t_vec_output = beam.t_vec_output*1e3

    # The coefficient depends on the magnitude of the time step (in this case milliseconds)
    interval = 1e6 * beam.output_frequency * dt

    t0_implicit_midpoint = time.perf_counter()
    dict_results_implicit_midpoint = beam.implicit_method(save_vars=True, type="implicit midpoint")
    tf_implicit_midpoint = time.perf_counter()

    t0_discrete_gradient = time.perf_counter()
    dict_results_discrete_gradient = beam.implicit_method(save_vars=True, type="discrete gradient")
    tf_discrete_gradient = time.perf_counter()

    # t0_leapfrog = time.perf_counter()
    # dict_results_leapfrog = beam.leapfrog(save_vars=True)
    # tf_leapfrog = time.perf_counter()

    energy_vec_imp_midpoint = dict_results_implicit_midpoint["energy"]
    q_x_list_imp_midpoint = dict_results_implicit_midpoint["hor_displacement"]
    q_x_array_imp_midpoint = beam.convert_functions_to_array(q_x_list_imp_midpoint)
    q_z_list_imp_midpoint = dict_results_implicit_midpoint["ver_displacement"]
    q_z_array_imp_midpoint = beam.convert_functions_to_array(q_z_list_imp_midpoint)
    

    energy_vec_dis_gradient = dict_results_discrete_gradient["energy"]
    q_x_list_dis_gradient = dict_results_discrete_gradient["hor_displacement"]
    q_x_array_dis_gradient = beam.convert_functions_to_array(q_x_list_dis_gradient)
    q_z_list_dis_gradient = dict_results_discrete_gradient["ver_displacement"]
    q_z_array_dis_gradient = beam.convert_functions_to_array(q_z_list_dis_gradient)
    
    plt.figure()
    plt.plot(t_vec_output, energy_vec_imp_midpoint, label="Imp midpoint")
    plt.plot(t_vec_output, energy_vec_dis_gradient, label="Dis gradient")
    plt.legend()
    plt.xlabel("Time [ms]")
    plt.title("Energy")

    hor_disp_at_point_imp_midpoint = q_x_array_imp_midpoint[:, index_point]
    hor_disp_at_point_dis_gradient = q_x_array_dis_gradient[:, index_point]

    plt.figure()
    plt.plot(t_vec_output, hor_disp_at_point_imp_midpoint, label="Imp midpoint")
    plt.plot(t_vec_output, hor_disp_at_point_dis_gradient, label="Dis gradient")
    plt.legend()
    plt.xlabel("Time [ms]")
    plt.title("Horizontal displacement")

    ver_disp_at_point_imp_midpoint = q_z_array_imp_midpoint[:, index_point]
    ver_disp_at_point_dis_gradient = q_z_array_dis_gradient[:, index_point]

    plt.figure()
    plt.plot(t_vec_output, ver_disp_at_point_imp_midpoint, label="Imp midpoint")
    plt.plot(t_vec_output, ver_disp_at_point_dis_gradient, label="Dis gradient")
    plt.legend()
    plt.xlabel("Time [ms]")
    plt.title("Vertical displacement")

    # anim_x_imp_midpoint = create_1d_line_animation(t_vec_output, x_vec, \
    #                         q_x_array_imp_midpoint, interval=interval,
    #                         xlabel="x", ylabel="$q_x$", \
    #                         title=f"Horizontal displacement $\\alpha={alpha}$", \
    #                         filename=f"{directory_results}Hor_disp_impl_midpoint.mp4")

    # fig_x_imp_midpoint, ax_x_imp_midpoint = plot_surface_from_matrix(t_vec_output, x_vec, \
    #                                             q_x_array_imp_midpoint, \
    #                                             x_label="$x$", y_label="$t$", z_label="$q_x$", \
    #                                             title="Horizontal displacement")

    # anim_z_imp_midpoint = create_1d_line_animation(t_vec_output, x_vec, \
    #                                             q_z_array_imp_midpoint, interval=interval/10, \
    #                                             xlabel="x", ylabel="$q_z$", \
    #                                             title=f"Vertical displacement $\\alpha={alpha}$", \
    #                                             filename=f"{directory_results}Ver_disp_impl_midpoint.mp4")

    # fig_z_imp_midpoint, ax_z_imp_midpoint = plot_surface_from_matrix(t_vec_output, x_vec, \
    #                                         q_z_array_imp_midpoint, \
    #                                         x_label="$x \; \mathrm{[m]}$", \
    #                                         y_label="$t \; \mathrm{[ms]}$", \
    #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
    #                                         title="Vertical displacement")
    
    
    
    plt.show()