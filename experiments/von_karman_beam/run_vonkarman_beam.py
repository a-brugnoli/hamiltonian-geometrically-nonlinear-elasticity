import firedrake as fdrk
import time
from von_karman_beam import VonKarmanBeam
import numpy as np
import matplotlib.pyplot as plt
from src.postprocessing.animation_1d_line import create_1d_line_animation
from src.postprocessing.plot_convergence import plot_convergence
from src.postprocessing.plot_surface import plot_surface_from_matrix
from src.norm_computation import error_norm, firedrake_error_norm
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

# # # If time step is set first, there is a minimum mesh size to guarantee stability
# # mesh_size_min_bending = np.sqrt(2*dt*wave_speed_bending)
# # mesh_size_min_traction = dt*wave_speed_traction
# # mesh_size_min = max(mesh_size_min_bending, mesh_size_min_traction)

# # mesh_size = sec_coeff*mesh_size_min
# # Given a number of elements the length is determined a posteriori
# # L = mesh_size*n_elements

# # Here we set the mesh size and the time step follows

dt_max_bending = mesh_size**2/(2*wave_speed_bending)
dt_max_traction = mesh_size/wave_speed_traction

dt_CFL_bending = dt_max_bending/sec_coeff_bend
dt_CFL_traction = dt_max_traction/sec_coeff_traction

# # Initial condition
alpha = 10
ampl_hor_disp_0 = 0 # alpha*height
ampl_ver_disp_0 = alpha*height

# # Time step to have sufficient resolution
dt_base = dt_CFL_bending
omega1_bending = (pi/L)**2*wave_speed_bending
T1_bending = 2*pi/omega1_bending

t_end_approx = T1_bending/10
n_steps_approx = np.round(t_end_approx/dt_base)

if n_steps_approx<1e4:
    n_sim_output = n_steps_approx
    n_steps = n_steps_approx
else:
    n_sim_output = 1000
    n_steps = np.round(n_steps_approx/n_sim_output)*n_sim_output

t_end = n_steps.astype(int)*dt_base
t_span = [0, t_end]

print(f"T end approx {t_end_approx}")
print(f"T end {t_end}")

n_case = 3
log_base = 2
time_step_vec = np.array([dt_base/log_base**n for n in range(n_case)])

# error_vec_q_leapfrog = np.zeros(n_case)
# error_vec_v_leapfrog = np.zeros(n_case)
# error_vec_E_leapfrog = np.zeros(n_case)
# elapsed_vec_leapfrog = np.zeros(n_case)

error_vec_q_x_dis_gradient = np.zeros(n_case)
error_vec_q_z_dis_gradient = np.zeros(n_case)
error_vec_v_x_dis_gradient = np.zeros(n_case)
error_vec_v_z_dis_gradient = np.zeros(n_case)
error_vec_E_dis_gradient = np.zeros(n_case)
elapsed_vec_dis_gradient = np.zeros(n_case)

error_vec_q_x_lin_implicit = np.zeros(n_case)
error_vec_q_z_lin_implicit = np.zeros(n_case)
error_vec_v_x_lin_implicit = np.zeros(n_case)
error_vec_v_z_lin_implicit = np.zeros(n_case)
error_vec_E_lin_implicit = np.zeros(n_case)
elapsed_vec_lin_implicit = np.zeros(n_case)


coeff_reference = 10
dt_reference = time_step_vec[-1]/coeff_reference

beam = VonKarmanBeam(time_step=dt_reference, t_span=t_span, n_output= n_sim_output,\
                        n_elem = n_elements, q0_hor = ampl_hor_disp_0, q0_ver=ampl_ver_disp_0, \
                        rho = rho, E = E, I = I, A=A, L=L)

t_vec_output_reference = beam.t_vec_output*1e3


print(f"Running reference")
t0_reference = time.perf_counter()
dict_results_reference = beam.leapfrog(save_vars=True)
tf_reference = time.perf_counter()

energy_vec_reference = dict_results_reference["energy"]
q_x_list_reference = dict_results_reference["horizontal displacement"]
q_z_list_reference = dict_results_reference["vertical displacement"]
v_x_list_reference = dict_results_reference["horizontal velocity"]
v_z_list_reference = dict_results_reference["vertical velocity"]


norm_type = "L2"

for ii in range(n_case):
    dt = time_step_vec[ii]
    print(f"Time step: {dt}")

    # beam = VonKarmanBeam(time_step=dt, t_span=t_span, n_output= n_sim_output,\
    #                     n_elem = n_elements, q0_hor = ampl_hor_disp_0, q0_ver=ampl_ver_disp_0, \
    #                     rho = rho, E = E, I = I, A=A, L=L)

    beam.set_time_step(dt)
    t_vec_output = beam.t_vec_output*1e3

    # print(f"T vector output after changing dt")
    # print(t_vec_output[:10])
    # print(t_vec_output.shape)

    
    x_vec = beam.x_vec
    x_point = L/8

    index_point = np.argmin(np.abs(x_vec - x_point))

    # The coefficient depends on the magnitude of the time step (in this case milliseconds)
    interval = 1e6 * beam.output_frequency * dt

    # print(f"Running implicit midpoint")
    # t0_imp_midpoint = time.perf_counter()
    # dict_results_imp_midpoint = beam.imp_method(save_vars=True, type="implicit midpoint")
    # tf_imp_midpoint = time.perf_counter()

    # energy_vec_imp_midpoint = dict_results_imp_midpoint["energy"]
    # q_x_list_imp_midpoint = dict_results_imp_midpoint["horizontal displacement"]
    # q_x_array_imp_midpoint = beam.convert_functions_to_array(q_x_list_imp_midpoint)
    # q_z_list_imp_midpoint = dict_results_imp_midpoint["vertical displacement"]
    # q_z_array_imp_midpoint = beam.convert_functions_to_array(q_z_list_imp_midpoint)


    print(f"Running discrete gradient")
    t0_dis_gradient = time.perf_counter()
    dict_results_dis_gradient = beam.implicit_method(save_vars=True, type="discrete gradient")
    tf_dis_gradient = time.perf_counter()

    energy_vec_dis_gradient = dict_results_dis_gradient["energy"]
    q_x_list_dis_gradient = dict_results_dis_gradient["horizontal displacement"]
    q_z_list_dis_gradient = dict_results_dis_gradient["vertical displacement"]
    v_x_list_dis_gradient = dict_results_dis_gradient["horizontal velocity"]
    v_z_list_dis_gradient = dict_results_dis_gradient["vertical velocity"]


    print(f"Running linearly implicit")
    t0_lin_implicit = time.perf_counter()
    dict_results_lin_implicit = beam.linear_implicit(save_vars=True)
    tf_lin_implicit = time.perf_counter()

    energy_vec_lin_implicit = dict_results_lin_implicit["energy"]
    q_x_list_lin_implicit = dict_results_lin_implicit["horizontal displacement"]
    q_z_list_lin_implicit = dict_results_lin_implicit["vertical displacement"]
    v_x_list_lin_implicit = dict_results_lin_implicit["horizontal velocity"]
    v_z_list_lin_implicit = dict_results_lin_implicit["vertical velocity"]

    elapsed_dis_gradient = (tf_dis_gradient - t0_dis_gradient)*1e3
    elapsed_lin_implicit = (tf_lin_implicit - t0_lin_implicit)*1e3

    print(f"Elapsed time Midpoint Discrete gradient [ms]: {elapsed_dis_gradient}")
    print(f"Elapsed time Linear implicit [ms]: {elapsed_lin_implicit}")

    
    # # Compute error
    error_q_x_dis_gradient = firedrake_error_norm(q_x_list_reference, q_x_list_dis_gradient, dt, norm=norm_type)
    error_q_x_lin_implicit = firedrake_error_norm(q_x_list_reference, q_x_list_lin_implicit, dt, norm=norm_type)

    error_q_z_dis_gradient = firedrake_error_norm(q_z_list_reference, q_z_list_dis_gradient, dt, norm=norm_type)
    error_q_z_lin_implicit = firedrake_error_norm(q_z_list_reference, q_z_list_lin_implicit, dt, norm=norm_type)

    error_v_x_dis_gradient = firedrake_error_norm(v_x_list_reference, v_x_list_dis_gradient, dt, norm=norm_type)
    error_v_x_lin_implicit = firedrake_error_norm(v_x_list_reference, v_x_list_lin_implicit, dt, norm=norm_type)

    error_v_z_dis_gradient = firedrake_error_norm(v_z_list_reference, v_z_list_dis_gradient, dt, norm=norm_type)
    error_v_z_lin_implicit = firedrake_error_norm(v_z_list_reference, v_z_list_lin_implicit, dt, norm=norm_type)

    error_vec_q_x_dis_gradient[ii] = error_q_x_dis_gradient
    error_vec_q_z_dis_gradient[ii] = error_q_z_dis_gradient

    error_vec_v_x_dis_gradient[ii] = error_v_x_dis_gradient
    error_vec_v_z_dis_gradient[ii] = error_v_z_dis_gradient

    # error_vec_E_dis_gradient[ii] = error_E_dis_gradient
    elapsed_vec_dis_gradient[ii] = elapsed_dis_gradient

    error_vec_q_x_lin_implicit[ii] = error_q_x_lin_implicit
    error_vec_q_z_lin_implicit[ii] = error_q_z_lin_implicit

    error_vec_v_x_lin_implicit[ii] = error_v_x_lin_implicit
    error_vec_v_z_lin_implicit[ii] = error_v_z_lin_implicit

    # error_vec_E_lin_implicit[ii] = error_E_lin_implicit
    elapsed_vec_lin_implicit[ii] = elapsed_lin_implicit


#     # q_x_array_dis_gradient = beam.convert_functions_to_array(q_x_list_dis_gradient)
#     # q_z_array_dis_gradient = beam.convert_functions_to_array(q_z_list_dis_gradient)

#     # q_x_array_lin_implicit = beam.convert_functions_to_array(q_x_list_lin_implicit)
#     # q_z_array_lin_implicit = beam.convert_functions_to_array(q_z_list_lin_implicit)

#     # print(f"Running Leapfrog")
#     # t0_leapfrog = time.perf_counter()
#     # dict_results_leapfrog = beam.leapfrog(save_vars=True)
#     # tf_leapfrog = time.perf_counter()

#     # energy_vec_leapfrog = dict_results_leapfrog["energy"]
#     # q_x_list_leapfrog = dict_results_leapfrog["horizontal displacement"]
#     # q_x_array_leapfrog = beam.convert_functions_to_array(q_x_list_leapfrog)
#     # q_z_list_leapfrog = dict_results_leapfrog["vertical displacement"]
#     # q_z_array_leapfrog = beam.convert_functions_to_array(q_z_list_leapfrog)
    
#     # plt.figure()
#     # plt.plot(t_vec_output, energy_vec_imp_midpoint, label="Implicit midpoint")
#     # plt.plot(t_vec_output, energy_vec_dis_gradient, label="Dis gradient")
#     # plt.plot(t_vec_output, energy_vec_lin_implicit, label="Linear implicit")
#     # plt.plot(t_vec_output, energy_vec_leapfrog, label="Leapfrog")
#     # plt.legend()
#     # plt.xlabel("Time [ms]")
#     # plt.title("Energy")

#     # hor_disp_at_point_imp_midpoint = q_x_array_imp_midpoint[:, index_point]
#     # hor_disp_at_point_dis_gradient = q_x_array_dis_gradient[:, index_point]
#     # hor_disp_at_point_lin_implicit = q_x_array_lin_implicit[:, index_point]
#     # hor_disp_at_point_leapfrog = q_x_array_leapfrog[:, index_point]

#     # plt.figure()
#     # plt.plot(t_vec_output, hor_disp_at_point_imp_midpoint, label="Implicit midpoint")
#     # plt.plot(t_vec_output, hor_disp_at_point_dis_gradient, label="Dis gradient")
#     # plt.plot(t_vec_output, hor_disp_at_point_lin_implicit, label="Linear implicit")
#     # plt.plot(t_vec_output, hor_disp_at_point_leapfrog, label="Leapfrog")
#     # plt.legend()
#     # plt.xlabel("Time [ms]")
#     # plt.title("Horizontal displacement")

#     # ver_disp_at_point_imp_midpoint = q_z_array_imp_midpoint[:, index_point]
#     # ver_disp_at_point_dis_gradient = q_z_array_dis_gradient[:, index_point]
#     # ver_disp_at_point_lin_implicit = q_z_array_lin_implicit[:, index_point]
#     # ver_disp_at_point_leapfrog = q_z_array_leapfrog[:, index_point]

#     # plt.figure()
#     # plt.plot(t_vec_output, ver_disp_at_point_imp_midpoint, label="Implicit midpoint")
#     # plt.plot(t_vec_output, ver_disp_at_point_dis_gradient, label="Dis gradient")
#     # plt.plot(t_vec_output, ver_disp_at_point_lin_implicit, label="Linear implicit")
#     # plt.plot(t_vec_output, ver_disp_at_point_leapfrog, label="Leapfrog")

#     # plt.legend()
#     # plt.xlabel("Time [ms]")
#     # plt.title("Vertical displacement")

#     # fig_x_imp_midpoint, ax_x_imp_midpoint = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                             q_x_array_imp_midpoint, \
#     #                                             x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                             title="Horizontal displacement discrete gradient")

    
#     # fig_z_imp_midpoint, ax_z_imp_midpoint = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_imp_midpoint, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement discrete gradient")


    
#     # fig_x_lin_implicit, ax_x_lin_implicit = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_x_array_lin_implicit, \
#     #                                         x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                         title="Horizontal displacement linear implicit")

    
#     # fig_z_lin_implicit, ax_z_lin_implicit = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_lin_implicit, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement lin implicit")
    


#     # fig_x_dis_gradient, ax_x_dis_gradient = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                             q_x_array_dis_gradient, \
#     #                                             x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                             title="Horizontal displacement discrete gradient")

    
#     # fig_z_dis_gradient, ax_z_dis_gradient = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_dis_gradient, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement discrete gradient")


#     # fig_x_leapfrog, ax_x_leapfrog = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                             q_x_array_leapfrog, \
#     #                                             x_label="$x$", y_label="$t$", z_label="$q_x$", \
#     #                                             title="Horizontal displacement leapfrog")

    
#     # fig_z_leapfrog, ax_z_leapfrog = plot_surface_from_matrix(t_vec_output, x_vec, \
#     #                                         q_z_array_leapfrog, \
#     #                                         x_label="$x \; \mathrm{[m]}$", \
#     #                                         y_label="$t \; \mathrm{[ms]}$", \
#     #                                         z_label="$q_z \; \mathrm{[m]}$ ", \
#     #                                         title="Vertical displacement leapfrog")    
    
#     # anim_x_imp_midpoint = create_1d_line_animation(t_vec_output, x_vec, \
#     #                         q_x_array_imp_midpoint, interval=interval,
#     #                         xlabel="x", ylabel="$q_x$", \
#     #                         title=f"Horizontal displacement $\\alpha={alpha}$", \
#     #                         filename=f"{directory_results}Hor_disp_impl_midpoint.mp4")

#     # anim_z_imp_midpoint = create_1d_line_animation(t_vec_output, x_vec, \
#     #                                             q_z_array_imp_midpoint, interval=interval/10, \
#     #                                             xlabel="x", ylabel="$q_z$", \
#     #                                             title=f"Vertical displacement $\\alpha={alpha}$", \
#     #                                             filename=f"{directory_results}Ver_disp_impl_midpoint.mp4")

    

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)

dict_hor_position = {"Discrete gradient": error_vec_q_x_dis_gradient,\
                "Linear implicit": error_vec_q_x_lin_implicit}

dict_ver_position = {"Discrete gradient": error_vec_q_z_dis_gradient,\
                "Linear implicit": error_vec_q_z_lin_implicit}


dict_hor_velocity = {"Discrete gradient": error_vec_v_x_dis_gradient,\
                "Linear implicit": error_vec_v_x_lin_implicit}

dict_ver_velocity = {"Discrete gradient": error_vec_v_z_dis_gradient,\
                "Linear implicit": error_vec_v_z_lin_implicit}


str_xlabel = '$\log \Delta t \; \mathrm{[s]}$'
plot_convergence(time_step_vec, dict_hor_position, xlabel=str_xlabel, ylabel="$\log \Delta q$", \
                title='Position error', savefig=f"{directory_results}convergence_horizontal_position.pdf")
plot_convergence(time_step_vec, dict_hor_velocity, xlabel=str_xlabel, ylabel="$\log \Delta v$",  \
                 title='Velocity error', savefig=f"{directory_results}convergence_horizontal_velocity.pdf")

plot_convergence(time_step_vec, dict_ver_position, xlabel=str_xlabel, ylabel="$\log \Delta q$", \
                title='Position error', savefig=f"{directory_results}convergence_vertical_position.pdf")
plot_convergence(time_step_vec, dict_ver_velocity, xlabel=str_xlabel, ylabel="$\log \Delta v$",  \
                 title='Velocity error', savefig=f"{directory_results}convergence_vertical_velocity.pdf")


plt.figure()
# # plt.loglog(time_step_vec, error_vec_E_leapfrog, '*-', label='Leapfrog')
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
# # plt.loglog(time_step_vec, elapsed_vec_leapfrog, '*-', label='Leapfrog')
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
