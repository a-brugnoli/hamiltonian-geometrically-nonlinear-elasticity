from experiments.finite_strain_elasticity.parameters_elasticity import *
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from src.postprocessing.options import configure_matplotib
configure_matplotib()

# print("Running Leapfrog")
# t0_leapfrog = time.perf_counter()
# dict_results_leapfrog = bending_column.leapfrog(save_vars=True)
# tf_leapfrog = time.perf_counter()

# disp_array_leapfrog = dict_results_leapfrog["displacement"]
# vel_array_leapfrog = dict_results_leapfrog["velocity"]
# energy_array_leapfrog = dict_results_leapfrog["energy"]

# disp_at_point_leapfrog = disp_array_leapfrog[:, index_point, :]

# t0_dis_gradient = time.perf_counter()
# dict_results_dis_gradient = bending_column.implicit_method(method="discrete gradient", save_vars=True)
# tf_dis_gradient = time.perf_counter()

# disp_array_dis_gradient = dict_results_dis_gradient["displacement"]
# vel_array_dis_gradient = dict_results_dis_gradient["velocity"]

# disp_at_point_dis_gradient = disp_array_dis_gradient[:, index_point, :]

# This is just to verify the equivalence of the two methods
t0_lin_implicit_strain = time.perf_counter()
dict_results_lin_implicit_strain = bending_column.linear_implicit_strain(save_vars=True,
                                                                    paraview_directory=paraview_directory)
tf_lin_implicit_strain = time.perf_counter()

disp_array_lin_implicit_strain = dict_results_lin_implicit_strain["displacement"]
vel_array_lin_implicit_strain = dict_results_lin_implicit_strain["velocity"]
energy_array_lin_implicit_strain = dict_results_lin_implicit_strain["energy"]

disp_at_point_lin_implicit_strain = disp_array_lin_implicit_strain[:, index_point, :]

# print("Running semiexplicit")
# t0_semiexplicit_strain = time.perf_counter()
# dict_results_semiexplicit_strain = bending_column.semiexplicit_strain_static_condensation(save_vars=True, 
#                                                                     paraview_directory=paraview_directory)
# tf_semiexplicit_strain = time.perf_counter()

# disp_array_semiexplicit_strain = dict_results_semiexplicit_strain["displacement"]
# vel_array_semiexplicit_strain = dict_results_semiexplicit_strain["velocity"]
# energy_array_semiexplicit_strain = dict_results_semiexplicit_strain["energy"]

# disp_at_point_semiexplicit_strain = disp_array_semiexplicit_strain[:, index_point, :]

print("Running linearly implicit")
t0_lin_implicit = time.perf_counter()
dict_results_lin_implicit = bending_column.linear_implicit_static_condensation(save_vars=True,
                                                                    paraview_directory=paraview_directory)
tf_lin_implicit = time.perf_counter()

disp_array_lin_implicit = dict_results_lin_implicit["displacement"]
vel_array_lin_implicit = dict_results_lin_implicit["velocity"]
energy_array_lin_implicit = dict_results_lin_implicit["energy"]

disp_at_point_lin_implicit = disp_array_lin_implicit[:, index_point, :]


diff_at_point_x = disp_at_point_lin_implicit[:, 0] - disp_at_point_lin_implicit_strain[:, 0]
diff_at_point_y = disp_at_point_lin_implicit[:, 1] - disp_at_point_lin_implicit_strain[:, 1]
diff_at_point_z = disp_at_point_lin_implicit[:, 2] - disp_at_point_lin_implicit_strain[:, 2]
plt.plot(t_vec_output_ms, diff_at_point_x, label="$\Delta q_x$")
plt.plot(t_vec_output_ms, diff_at_point_y, label="$\Delta q_y$")
plt.plot(t_vec_output_ms, diff_at_point_z, label="$\Delta q_z$")
plt.xlabel("Time [ms]")
plt.title(r"Difference Displacement at Point $\bm{x}=[0, 0, L_z]$")
plt.legend()

plt.figure()
plt.plot(t_vec_output_ms, energy_array_lin_implicit-energy_array_lin_implicit_strain)
plt.xlabel("Time [ms]")
plt.title("Difference Energy")
plt.legend()

plt.show()

# plt.figure()
# plt.plot(t_vec_output_ms, disp_at_point_leapfrog[:, 0], label="LP x")
# plt.plot(t_vec_output_ms, disp_at_point_dis_gradient[:, 0], label="DG x")
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit[:, 0], label="LI x")
# plt.plot(t_vec_output_ms, disp_at_point_semiexplicit_strain[:, 0], label="SE x")

# To show equivalence with the strain linear implicit method
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit_strain[:, 0], label="LI strain x")
# plt.xlabel("Time [ms]")
# plt.title("Displacement x")
# plt.legend()

# plt.figure()
# plt.plot(t_vec_output_ms, disp_at_point_leapfrog[:, 1], label="LP y")
# plt.plot(t_vec_output_ms, disp_at_point_dis_gradient[:, 1], label="DG y")
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit[:, 1], label="LI y")
# plt.plot(t_vec_output_ms, disp_at_point_semiexplicit_strain[:, 1], label="SE y")

# # To show equivalence with the strain linear implicit method
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit_strain[:, 1], label="LI strain y")
# plt.xlabel("Time [ms]")
# plt.title("Displacement y")
# plt.legend()

# plt.figure()
# plt.plot(t_vec_output_ms, disp_at_point_leapfrog[:, 2], label="LP z")
# plt.plot(t_vec_output_ms, disp_at_point_dis_gradient[:, 2], label="DG z")
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit[:, 2], label="LI z")
# plt.plot(t_vec_output_ms, disp_at_point_semiexplicit_strain[:, 2], label="SE z")

# # To show equivalence with the strain linear implicit method
# plt.plot(t_vec_output_ms, disp_at_point_lin_implicit_strain[:, 2], label="LI stress z")
# plt.xlabel("Time [ms]")
# plt.title("Displacement z")
# plt.legend()

# plt.figure()
# plt.plot(t_vec_output_ms, energy_array_leapfrog, label="LF")
# plt.plot(t_vec_output_ms, energy_array_lin_implicit_strain, label="Strain")
# plt.plot(t_vec_output_ms, energy_array_semiexplicit_strain, label="SE")
# plt.plot(t_vec_output_ms, energy_array_lin_implicit, label="LI")
# plt.xlabel("Time [ms]")
# plt.title("Energy")
# plt.legend()

# plt.show()