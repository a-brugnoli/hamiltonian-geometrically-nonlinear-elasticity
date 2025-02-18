import time
import pickle
from experiments.vonkarman_beam.vonkarman_beam import VonKarmanBeam
import numpy as np
import matplotlib.pyplot as plt
from src.postprocessing.animation_1d_line import create_1d_line_animation
from src.postprocessing.plot_convergence import plot_convergence
from src.postprocessing.plot_surface import plot_surface_from_matrix
from math import pi
import os
from parameters import *

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)
n_case = 3
log_base = 2
time_step_vec = np.array([dt_base/log_base**n for n in range(n_case)])

coeff_reference = 2**7
dt_reference = time_step_vec[0]/coeff_reference

beam = VonKarmanBeam(time_step=dt_reference, t_span=t_span, n_output= n_sim_output,\
                        n_elem = n_elements, q0_hor = ampl_hor_disp_0, q0_ver=ampl_ver_disp_0, \
                        rho = rho, E = E, I = I, A=A, L=L)

x_vec = beam.x_vec
x_point = L/4
index_point = np.argmin(np.abs(x_vec - x_point))

t_vec_output_reference = beam.t_vec_output
t_vec_output_ms_reference = t_vec_output_reference*1e3

dt_output = np.mean(np.diff(t_vec_output_reference)) 

file_results_reference = directory_results + "results_reference.pkl"

try:
    with open(file_results_reference, "rb") as f:
        dict_results_reference = pickle.load(f)
    print("Dictionary loaded successfully")
except FileNotFoundError:
    print(f"Error: The file '{file_results_reference}' does not exist.")
    print(f"Running reference")
    t0_reference = time.perf_counter()
    dict_results_reference = beam.leapfrog(save_vars=True)
    tf_reference = time.perf_counter()
    with open(file_results_reference, "wb") as f:
        pickle.dump(dict_results_reference, f)

except pickle.UnpicklingError:
    print("Error: The file contains invalid or corrupted data.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


energy_vec_reference = dict_results_reference["energy"]
q_x_array_reference = dict_results_reference["horizontal displacement"]
q_z_array_reference = dict_results_reference["vertical displacement"]
v_x_array_reference = dict_results_reference["horizontal velocity"]
v_z_array_reference = dict_results_reference["vertical velocity"]

hor_disp_at_point_reference = q_x_array_reference[:, index_point]
ver_disp_at_point_reference = q_z_array_reference[:, index_point]

results_energy_vec_dis_gradient = np.array((n_sim_output, n_case))
diff_E_vec_dis_gradient = np.zeros(n_case)

results_q_x_array_dis_gradient = np.array((n_sim_output, n_dofs_hor, n_case))
results_v_x_array_dis_gradient = np.array((n_sim_output, n_dofs_hor, n_case))
results_q_z_array_dis_gradient = np.array((n_sim_output, n_dofs_ver, n_case))
results_v_z_array_dis_gradient = np.array((n_sim_output, n_dofs_ver, n_case))
elapsed_vec_dis_gradient = np.array(n_case)


results_energy_vec_lin_implicit = np.array((n_sim_output, n_case))
diff_E_vec_lin_implicit = np.zeros(n_case)

results_q_x_array_lin_implicit = np.array((n_sim_output, n_dofs_hor, n_case))
results_v_x_array_lin_implicit = np.array((n_sim_output, n_dofs_hor, n_case))
results_q_z_array_lin_implicit = np.array((n_sim_output, n_dofs_ver, n_case))
results_v_z_array_lin_implicit = np.array((n_sim_output, n_dofs_ver, n_case))
elapsed_vec_lin_implicit = np.array(n_case)


for ii in range(n_case):
    dt = time_step_vec[ii]
    print(f"Time step: {dt}")

    # beam = VonKarmanBeam(time_step=dt, t_span=t_span, n_output= n_sim_output,\
    #                     n_elem = n_elements, q0_hor = ampl_hor_disp_0, q0_ver=ampl_ver_disp_0, \
    #                     rho = rho, E = E, I = I, A=A, L=L)

    beam.set_time_step(dt)
    t_vec_output = beam.t_vec_output
    t_vec_output_ms = t_vec_output*1e3

    print(f"T output diff : {np.linalg.norm(t_vec_output_reference-t_vec_output)}")

#     # The coefficient depends on the magnitude of the time step (in this case milliseconds)
#     interval = 1e6 * beam.output_frequency * dt

    print(f"Running discrete gradient")
    t0_dis_gradient = time.perf_counter()
    dict_results_dis_gradient = beam.implicit_method(save_vars=True, type="discrete gradient")
    tf_dis_gradient = time.perf_counter()

    results_energy_vec_dis_gradient[:, ii] = dict_results_dis_gradient["energy"]
    results_q_x_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["horizontal displacement"]
    results_v_x_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["horizontal velocity"]
    results_q_z_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["vertical displacement"]
    results_v_z_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["vertical velocity"]

    print(f"Running linearly implicit")
    t0_lin_implicit = time.perf_counter()
    dict_results_lin_implicit = beam.linear_implicit(save_vars=True)
    tf_lin_implicit = time.perf_counter()

    results_energy_vec_lin_implicit[:, ii] = dict_results_lin_implicit["energy"]
    results_q_x_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["horizontal displacement"]
    results_v_x_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["horizontal velocity"]
    results_q_z_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["vertical displacement"]
    results_v_z_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["vertical velocity"]

    elapsed_vec_dis_gradient[ii] = tf_dis_gradient - t0_dis_gradient
    elapsed_vec_lin_implicit[ii] = tf_lin_implicit - t0_lin_implicit

    print(f"Elapsed time Midpoint Discrete gradient : {elapsed_vec_dis_gradient[ii]}")
    print(f"Elapsed time Linear implicit : {elapsed_vec_lin_implicit[ii]}")

    
dict_results_dis_gradient = {"energy": results_energy_vec_dis_gradient, 
                             "horizontal displacement": results_q_x_array_dis_gradient,
                             "horizontal velocity": results_v_x_array_dis_gradient,
                             "vertical displacement": results_q_z_array_dis_gradient, 
                             "vertical velocity": results_v_z_array_dis_gradient}


dict_results_lin_implicit = {"energy": results_energy_vec_lin_implicit, 
                             "horizontal displacement": results_q_x_array_lin_implicit,
                             "horizontal velocity": results_v_x_array_lin_implicit,
                             "vertical displacement": results_q_z_array_lin_implicit, 
                             "vertical velocity": results_v_z_array_lin_implicit}
