import time
import pickle
import numpy as np
from experiments.vonkarman_beam.parameters_vonkarman import *

try:
    with open(file_results_reference, "rb") as f:
        dict_results_reference = pickle.load(f)
    print("Dictionary reference results loaded successfully")
except FileNotFoundError:
    print(f"Error: The file '{file_results_reference}' does not exist.")
    print(f"Running reference")
    beam.set_time_step(dt_reference)
    t0_reference = time.perf_counter()
    dict_results_reference = beam.leapfrog(save_vars=True)
    tf_reference = time.perf_counter()
    with open(file_results_reference, "wb") as f:
        pickle.dump(dict_results_reference, f)
except pickle.UnpicklingError:
    print(f"Error: The file {dict_results_reference} contains invalid or corrupted data.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    with open(file_results_linear, "rb") as f:
        dict_results_linear = pickle.load(f)
    print("Dictionary linear results loaded successfully")
except FileNotFoundError:
    print(f"Error: The file '{file_results_linear}' does not exist.")
    print(f"Running linear analysis")
    beam.set_time_step(dt_reference)
    t0_linear = time.perf_counter()
    dict_results_linear = beam.leapfrog(save_vars=True, linear=True)
    tf_linear = time.perf_counter()
    with open(file_results_linear, "wb") as f:
        pickle.dump(dict_results_linear, f)
except pickle.UnpicklingError:
    print(f"Error: The file {file_results_linear} contains invalid or corrupted data.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

n_dofs_hor_disp = beam.space_hor_disp.dim()

energy_vec_leapfrog = np.zeros((n_sim_output, n_cases))
q_x_array_leapfrog = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
v_x_array_leapfrog = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
q_z_array_leapfrog = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
v_z_array_leapfrog = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
elapsed_vec_leapfrog = np.zeros(n_cases)

energy_vec_dis_gradient = np.zeros((n_sim_output, n_cases))
q_x_array_dis_gradient = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
v_x_array_dis_gradient = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
q_z_array_dis_gradient = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
v_z_array_dis_gradient = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
elapsed_vec_dis_gradient = np.zeros(n_cases)

energy_vec_lin_implicit = np.zeros((n_sim_output, n_cases))
q_x_array_lin_implicit = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
v_x_array_lin_implicit = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
q_z_array_lin_implicit = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
v_z_array_lin_implicit = np.zeros((n_sim_output, n_dofs_hor_disp, n_cases))
elapsed_vec_lin_implicit = np.zeros(n_cases)


for ii in range(n_cases):
    dt = time_step_vec[ii]
    print(f"Time step: {dt}")

    beam.set_time_step(dt)
    
    assert np.linalg.norm(t_vec_output-beam.t_vec_output)<1e-12

    print(f"Running leapfrog")
    t0_leapfrog = time.perf_counter()
    dict_results_leapfrog = beam.leapfrog(save_vars=True)
    tf_leapfrog = time.perf_counter()
    elapsed_vec_leapfrog[ii] = tf_leapfrog - t0_leapfrog

    energy_vec_leapfrog[:, ii] = dict_results_leapfrog["energy"]
    q_x_array_leapfrog[:, :, ii] = dict_results_leapfrog["horizontal displacement"]
    v_x_array_leapfrog[:, :, ii] = dict_results_leapfrog["horizontal velocity"]
    q_z_array_leapfrog[:, :, ii] = dict_results_leapfrog["vertical displacement"]
    v_z_array_leapfrog[:, :, ii] = dict_results_leapfrog["vertical velocity"]


    print(f"Running discrete gradient")
    t0_dis_gradient = time.perf_counter()
    dict_results_dis_gradient = beam.implicit_method(method="discrete gradient", save_vars=True)
    tf_dis_gradient = time.perf_counter()
    elapsed_vec_dis_gradient[ii] = tf_dis_gradient - t0_dis_gradient

    energy_vec_dis_gradient[:, ii] = dict_results_dis_gradient["energy"]
    q_x_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["horizontal displacement"]
    v_x_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["horizontal velocity"]
    q_z_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["vertical displacement"]
    v_z_array_dis_gradient[:, :, ii] = dict_results_dis_gradient["vertical velocity"]

    print(f"Running linearly implicit")
    t0_lin_implicit = time.perf_counter()
    dict_results_lin_implicit = beam.linear_implicit(save_vars=True)
    tf_lin_implicit = time.perf_counter()
    elapsed_vec_lin_implicit[ii] = tf_lin_implicit - t0_lin_implicit

    energy_vec_lin_implicit[:, ii] = dict_results_lin_implicit["energy"]
    q_x_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["horizontal displacement"]
    v_x_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["horizontal velocity"]
    q_z_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["vertical displacement"]
    v_z_array_lin_implicit[:, :, ii] = dict_results_lin_implicit["vertical velocity"]

    print(f"Elapsed time Leapfrog : {elapsed_vec_leapfrog[ii]}")
    print(f"Elapsed time Midpoint Discrete gradient : {elapsed_vec_dis_gradient[ii]}")
    print(f"Elapsed time Linear implicit : {elapsed_vec_lin_implicit[ii]}")

dict_time = {"Time":t_vec_output}
with open(file_time, "wb") as f:
        pickle.dump(dict_time, f)
        
dict_results_leapfrog_cases = {"energy": energy_vec_leapfrog, 
                                "horizontal displacement": q_x_array_leapfrog,
                                "horizontal velocity": v_x_array_leapfrog,
                                "vertical displacement": q_z_array_leapfrog, 
                                "vertical velocity": v_z_array_leapfrog,
                                "elapsed time": elapsed_vec_leapfrog}

with open(file_results_leapfrog, "wb") as f:
        pickle.dump(dict_results_leapfrog_cases, f)



dict_results_dis_gradient_cases = {"energy": energy_vec_dis_gradient, 
                                "horizontal displacement": q_x_array_dis_gradient,
                                "horizontal velocity": v_x_array_dis_gradient,
                                "vertical displacement": q_z_array_dis_gradient, 
                                "vertical velocity": v_z_array_dis_gradient,
                                "elapsed time": elapsed_vec_dis_gradient}

with open(file_results_dis_gradient, "wb") as f:
        pickle.dump(dict_results_dis_gradient_cases, f)

dict_results_lin_implicit_cases = {"energy": energy_vec_lin_implicit, 
                                "horizontal displacement": q_x_array_lin_implicit,
                                "horizontal velocity": v_x_array_lin_implicit,
                                "vertical displacement": q_z_array_lin_implicit, 
                                "vertical velocity": v_z_array_lin_implicit,
                                "elapsed time": elapsed_vec_lin_implicit}


with open(file_results_lin_implicit, "wb") as f:
        pickle.dump(dict_results_lin_implicit_cases, f)

