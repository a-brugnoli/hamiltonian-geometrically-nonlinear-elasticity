from experiments.finite_strain_elasticity.parameters_elasticity import *
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from src.postprocessing.options import configure_matplotib
configure_matplotib()

t0_semiexplicit_static_cond = time.perf_counter()
dict_results_semiexplicit_static_cond = bending_column.semiexplicit_strain_static_condensation(save_vars=True)
tf_semiexplicit_static_cond = time.perf_counter()

q_array_semiexplicit_static_cond = dict_results_semiexplicit_static_cond["displacement"]
v_array_semiexplicit_static_cond = dict_results_semiexplicit_static_cond["velocity"]

disp_at_point_semiexplicit_static_cond = q_array_semiexplicit_static_cond[:, index_point, :]

t0_semiexplicit = time.perf_counter()
dict_results_semiexplicit = bending_column.semiexplicit_strain(save_vars=True)
tf_semiexplicit = time.perf_counter()

q_array_semiexplicit = dict_results_semiexplicit["displacement"]
v_array_semiexplicit = dict_results_semiexplicit["velocity"]

disp_at_point_semiexplicit = q_array_semiexplicit[:, index_point, :]

elapsed_vec_semiexplicit_static_cond = tf_semiexplicit_static_cond - t0_semiexplicit_static_cond
elapsed_vec_semiexplicit = tf_semiexplicit - t0_semiexplicit
print(f"Elapsed time Linear implicit static condensation : {elapsed_vec_semiexplicit_static_cond}")
print(f"Elapsed time Linear implicit : {elapsed_vec_semiexplicit}")

diff_at_point = disp_at_point_semiexplicit-disp_at_point_semiexplicit_static_cond
plt.figure()
plt.plot(t_vec_output_ms, diff_at_point[:, 0], label=fr"$\Delta q_x$")
plt.plot(t_vec_output_ms, diff_at_point[:, 1], label=fr"$\Delta q_y$")
plt.plot(t_vec_output_ms, diff_at_point[:, 2], label=fr"$\Delta q_z$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Displacement diff at point")

diff_q_array = q_array_semiexplicit_static_cond - q_array_semiexplicit
fig = plt.figure()
plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1)[:, 0], label="$\Delta q_x$")
plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1)[:, 1], label="$\Delta q_y$")
plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1)[:, 2], label="$\Delta q_z$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Euclidian norm Displacement diff")

diff_v_array = v_array_semiexplicit_static_cond - v_array_semiexplicit
fig = plt.figure()
plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1)[:, 0], label="$\Delta v_x$")
plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1)[:, 1], label="$\Delta v_y$")
plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1)[:, 2], label="$\Delta v_z$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Euclidian norm Velocity diff")
plt.show()