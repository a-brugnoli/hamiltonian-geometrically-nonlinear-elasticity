import time
import pickle
import numpy as np
from experiments.finite_strain_elasticity.parameters_elasticity import *

try:
    with open(file_results_reference, "rb") as f:
        dict_results_reference = pickle.load(f)
    print("Dictionary loaded successfully")
except FileNotFoundError:
    print(f"Error: The file '{file_results_reference}' does not exist.")
    print(f"Running reference")
    bending_column.set_time_step(dt_reference)
    t0_reference = time.perf_counter()
    dict_results_reference = bending_column.leapfrog(save_vars=True, paraview_directory=paraview_directory)
    tf_reference = time.perf_counter()
    with open(file_results_reference, "wb") as f:
        pickle.dump(dict_results_reference, f)
except pickle.UnpicklingError:
    print("Error: The file contains invalid or corrupted data.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

energy_vec_leapfrog = np.zeros((n_sim_output, n_cases))
q_array_leapfrog = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
v_array_leapfrog = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
elapsed_vec_leapfrog = np.zeros(n_cases)

energy_vec_dis_gradient = np.zeros((n_sim_output, n_cases))
q_array_dis_gradient = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
v_array_dis_gradient = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
elapsed_vec_dis_gradient = np.zeros(n_cases)

energy_vec_lin_implicit = np.zeros((n_sim_output, n_cases))
q_array_lin_implicit = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
v_array_lin_implicit = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
elapsed_vec_lin_implicit = np.zeros(n_cases)


for ii in range(n_cases):
    dt = time_step_vec[ii]
    print(f"Time step: {dt}")

    bending_column.set_time_step(dt)
    assert np.linalg.norm(t_vec_output-bending_column.t_vec_output)<1e-12

    print(f"Running leapfrog")
    t0_leapfrog = time.perf_counter()
    dict_results_leapfrog = bending_column.leapfrog(save_vars=True)
    tf_leapfrog = time.perf_counter()
    elapsed_vec_leapfrog[ii] = tf_leapfrog - t0_leapfrog

    energy_vec_leapfrog[:, ii] = dict_results_leapfrog["energy"]
    q_array_leapfrog[:, :, :, ii] = dict_results_leapfrog["displacement"]
    v_array_leapfrog[:, :, :, ii] = dict_results_leapfrog["velocity"]


    print(f"Running discrete gradient")
    t0_dis_gradient = time.perf_counter()
    dict_results_dis_gradient = bending_column.implicit_method(method="discrete gradient", save_vars=True)
    tf_dis_gradient = time.perf_counter()
    elapsed_vec_dis_gradient[ii] = tf_dis_gradient - t0_dis_gradient

    energy_vec_dis_gradient[:, ii] = dict_results_dis_gradient["energy"]
    q_array_dis_gradient[:, :, :, ii] = dict_results_dis_gradient["displacement"]
    v_array_dis_gradient[:, :, :, ii] = dict_results_dis_gradient["velocity"]

    print(f"Running linearly implicit")
    t0_lin_implicit = time.perf_counter()
    dict_results_lin_implicit = bending_column.linear_implicit_static_condensation(save_vars=True)
    tf_lin_implicit = time.perf_counter()
    elapsed_vec_lin_implicit[ii] = tf_lin_implicit - t0_lin_implicit

    energy_vec_lin_implicit[:, ii] = dict_results_lin_implicit["energy"]
    q_array_lin_implicit[:, :, :, ii] = dict_results_lin_implicit["displacement"]
    v_array_lin_implicit[:, :, :, ii] = dict_results_lin_implicit["velocity"]

    print(f"Elapsed time Leapfrog : {elapsed_vec_leapfrog[ii]}")
    print(f"Elapsed time Midpoint Discrete gradient : {elapsed_vec_dis_gradient[ii]}")
    print(f"Elapsed time Linear implicit : {elapsed_vec_lin_implicit[ii]}")


dict_time = {"Time":t_vec_output}
with open(file_time, "wb") as f:
        pickle.dump(dict_time, f)

dict_results_leapfrog_cases = {"energy": energy_vec_leapfrog, 
                                "displacement": q_array_leapfrog,
                                "velocity": v_array_leapfrog,
                                "elapsed time": elapsed_vec_leapfrog}

with open(file_results_leapfrog, "wb") as f:
        pickle.dump(dict_results_leapfrog_cases, f)


dict_results_dis_gradient_cases = {"energy": energy_vec_dis_gradient, 
                                "displacement": q_array_dis_gradient,
                                "velocity": v_array_dis_gradient,
                                "elapsed time": elapsed_vec_dis_gradient}

with open(file_results_dis_gradient, "wb") as f:
        pickle.dump(dict_results_dis_gradient_cases, f)

dict_results_lin_implicit_cases = {"energy": energy_vec_lin_implicit, 
                                "displacement": q_array_lin_implicit,
                                "velocity": v_array_lin_implicit,
                                "elapsed time": elapsed_vec_lin_implicit}


with open(file_results_lin_implicit, "wb") as f:
        pickle.dump(dict_results_lin_implicit_cases, f)

