from experiments.finite_strain_elasticity.parameters_elasticity import *
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from src.postprocessing.options import configure_matplotib
configure_matplotib()

t0_leapfrog = time.perf_counter()
dict_results_leapfrog = bending_column.leapfrog(save_vars=True)
tf_leapfrog = time.perf_counter()

disp_array_leapfrog = dict_results_leapfrog["displacement"]
vel_array_leapfrog = dict_results_leapfrog["velocity"]

disp_at_point_leapfrog = disp_array_leapfrog[:, index_point, :]

t0_dis_gradient = time.perf_counter()
dict_results_dis_gradient = bending_column.implicit_method(method="discrete gradient", save_vars=True)
tf_dis_gradient = time.perf_counter()

disp_array_dis_gradient = dict_results_dis_gradient["displacement"]
vel_array_dis_gradient = dict_results_dis_gradient["velocity"]

disp_at_point_dis_gradient = disp_array_dis_gradient[:, index_point, :]

t0_lin_implicit = time.perf_counter()
dict_results_lin_implicit = bending_column.linear_implicit_static_condensation(save_vars=True)
# dict_results_lin_implicit = bending_column.linear_implicit_static_condensation(save_vars=True)
tf_lin_implicit = time.perf_counter()

disp_array_lin_implicit = dict_results_lin_implicit["displacement"]
vel_array_lin_implicit = dict_results_lin_implicit["velocity"]

disp_at_point_lin_implicit = disp_array_lin_implicit[:, index_point, :]

plt.figure()
plt.plot(t_vec_output_ms, disp_at_point_leapfrog[:, 0], label="LP x")
plt.plot(t_vec_output_ms, disp_at_point_dis_gradient[:, 0], label="DG x")
plt.plot(t_vec_output_ms, disp_at_point_lin_implicit[:, 0], label="LI x")
plt.xlabel("Time [ms]")
plt.title("Displacement x")
plt.legend()

plt.figure()
plt.plot(t_vec_output_ms, disp_at_point_leapfrog[:, 1], label="LP y")
plt.plot(t_vec_output_ms, disp_at_point_dis_gradient[:, 1], label="DG y")
plt.plot(t_vec_output_ms, disp_at_point_lin_implicit[:, 1], label="LI y")
plt.xlabel("Time [ms]")
plt.title("Displacement x")
plt.legend()

plt.figure()
plt.plot(t_vec_output_ms, disp_at_point_leapfrog[:, 2], label="LP z")
plt.plot(t_vec_output_ms, disp_at_point_dis_gradient[:, 2], label="DG z")
plt.plot(t_vec_output_ms, disp_at_point_lin_implicit[:, 2], label="LI z")
plt.xlabel("Time [ms]")
plt.title("Displacement x")
plt.legend()

plt.show()