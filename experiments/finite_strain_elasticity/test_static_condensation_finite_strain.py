from experiments.finite_strain_elasticity.parameters_elasticity import *
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from src.postprocessing.options import configure_matplotib
configure_matplotib()


dt = time_step_vec[0]
bending_column.set_time_step(dt)

t0_lin_implicit_static_cond = time.perf_counter()
dict_results_lin_implicit_static_cond = bending_column.linear_implicit_static_condensation(save_vars=True)
tf_lin_implicit_static_cond = time.perf_counter()

q_array_lin_implicit_static_cond = dict_results_lin_implicit_static_cond["displacement"]
v_array_lin_implicit_static_cond = dict_results_lin_implicit_static_cond["velocity"]

disp_at_point_lin_implicit_static_cond = q_array_lin_implicit_static_cond[:, index_point, :]

# t0_lin_implicit = time.perf_counter()
# dict_results_lin_implicit = bending_column.linear_implicit(save_vars=True)
# tf_lin_implicit = time.perf_counter()

# q_array_lin_implicit = dict_results_lin_implicit["displacement"]
# v_array_lin_implicit = dict_results_lin_implicit["velocity"]

# disp_at_point_lin_implicit = q_array_lin_implicit[:, index_point, :]

# elapsed_vec_lin_implicit_static_cond = tf_lin_implicit_static_cond - t0_lin_implicit_static_cond
# elapsed_vec_lin_implicit = tf_lin_implicit - t0_lin_implicit
# print(f"Elapsed time Linear implicit static condensation : {elapsed_vec_lin_implicit_static_cond}")
# print(f"Elapsed time Linear implicit : {elapsed_vec_lin_implicit}")

# plt.figure()
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit, \
#             label=fr"Ref $\Delta t = {dt*1e6:.1f} \; \mathrm{{[\mu s]}}$")
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit_static_cond, \
#             label=fr"DG $\Delta t = {dt*1e6:.1f} \; \mathrm{{[\mu s]}}$")
# plt.legend()
# plt.xlabel("Time [ms]")
# plt.title("Vertical displacement")

# diff_q_array = q_array_lin_implicit_static_cond - q_array_lin_implicit
# diff_v_array = v_array_lin_implicit_static_cond - v_array_lin_implicit

# fig = plt.figure()
# plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1), label="$\Delta q$")
# plt.plot(t_vec_output_ms, np.linalg.norm(diff_q_array, axis=1), label="$\Delta $")

# plt.show()