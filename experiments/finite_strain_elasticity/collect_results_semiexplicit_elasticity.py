import time
import pickle
import numpy as np
from experiments.finite_strain_elasticity.parameters_elasticity import *

energy_vec_semiexplicit = np.zeros((n_sim_output, n_cases))
q_array_semiexplicit = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
v_array_semiexplicit = np.zeros((n_sim_output, n_dofs_disp, space_dim, n_cases))
elapsed_vec_semiexplicit = np.zeros(n_cases)

for ii in range(n_cases):
    dt = time_step_vec[ii]
    print(f"Time step: {dt}")

    bending_column.set_time_step(dt)
    assert np.linalg.norm(t_vec_output-bending_column.t_vec_output)<1e-12

    print(f"Running semiexplicit")
    t0_semiexplicit = time.perf_counter()
    dict_results_semiexplicit = bending_column.semiexplicit_strain_static_condensation(save_vars=True)
    tf_semiexplicit = time.perf_counter()
    elapsed_vec_semiexplicit[ii] = tf_semiexplicit - t0_semiexplicit

    energy_vec_semiexplicit[:, ii] = dict_results_semiexplicit["energy"]
    q_array_semiexplicit[:, :, :, ii] = dict_results_semiexplicit["displacement"]
    v_array_semiexplicit[:, :, :, ii] = dict_results_semiexplicit["velocity"]

    print(f"Elapsed time Semiexplicit : {elapsed_vec_semiexplicit[ii]}")


dict_time = {"Time":t_vec_output}
with open(file_time, "wb") as f:
        pickle.dump(dict_time, f)

dict_results_semiexplicit_cases = {"energy": energy_vec_semiexplicit, 
                                "displacement": q_array_semiexplicit,
                                "velocity": v_array_semiexplicit,
                                "elapsed time": elapsed_vec_semiexplicit}


with open(file_results_semiexplicit, "wb") as f:
        pickle.dump(dict_results_semiexplicit_cases, f)

