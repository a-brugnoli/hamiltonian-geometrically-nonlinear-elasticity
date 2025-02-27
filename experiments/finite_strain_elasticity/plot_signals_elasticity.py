import matplotlib.pyplot as plt
from experiments.finite_strain_elasticity.parameters_elasticity import *
import pickle
from src.postprocessing.options import configure_matplotib
configure_matplotib()

with open(file_time, "rb") as f:
        dict_time = pickle.load(f)

t_vec_results_ms = 1e3*dict_time["Time"]

with open(file_results_reference, "rb") as f:
        dict_results_reference = pickle.load(f)

energy_vec_reference = dict_results_reference["energy"]
q_array_reference = dict_results_reference["displacement"]
v_array_reference = dict_results_reference["velocity"]

with open(file_results_leapfrog, "rb") as f:
        dict_results_leapfrog = pickle.load(f)

energy_vec_leapfrog = dict_results_leapfrog["energy"]
q_array_leapfrog = dict_results_leapfrog["displacement"]
v_array_leapfrog = dict_results_leapfrog["velocity"]
comp_time_leapfrog = dict_results_leapfrog["elapsed time"]

diff_E_leapfrog = np.diff(energy_vec_leapfrog, axis=0)

with open(file_results_dis_gradient, "rb") as f:
        dict_results_dis_gradient = pickle.load(f)

energy_vec_dis_gradient = dict_results_dis_gradient["energy"]
q_array_dis_gradient = dict_results_dis_gradient["displacement"]
v_array_dis_gradient = dict_results_dis_gradient["velocity"]
comp_time_dis_gradient = dict_results_dis_gradient["elapsed time"]

diff_E_dis_gradient = np.diff(energy_vec_dis_gradient, axis=0)

with open(file_results_lin_implicit, "rb") as f:
        dict_results_lin_implicit = pickle.load(f)

energy_vec_lin_implicit = dict_results_lin_implicit["energy"]
q_array_lin_implicit = dict_results_lin_implicit["displacement"]
v_array_lin_implicit = dict_results_lin_implicit["velocity"]
comp_time_lin_implicit = dict_results_lin_implicit["elapsed time"]

diff_E_lin_implicit = np.diff(energy_vec_lin_implicit, axis=0)

plt.figure()
for ii in range(n_cases):
    plt.plot(t_vec_results_ms, energy_vec_dis_gradient[:, ii], '-.', \
             label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
    plt.plot(t_vec_results_ms, energy_vec_lin_implicit[:, ii], ':', \
             label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
    if mask_stable_leapfrog[ii]:
        plt.plot(t_vec_results_ms, energy_vec_leapfrog[:, ii], '-.', \
             label=fr"LF $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Energy")


plt.figure()
for ii in range(n_cases):
    plt.plot(t_vec_results_ms[1:], diff_E_dis_gradient[:, ii], '-.', \
            label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
    plt.plot(t_vec_results_ms[1:], diff_E_lin_implicit[:, ii], ':', \
            label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
    if mask_stable_leapfrog[ii]:
        plt.plot(t_vec_results_ms[1:], diff_E_leapfrog[:, ii], '-.', \
                label=fr"LF $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.ylabel("$H(t_{n+1}) - H(t_n)$")
plt.title("Difference Energy")

q_x_at_point_reference = q_array_reference[:, index_point, 0]
q_x_at_point_leapfrog = q_array_leapfrog[:, index_point, 0, :]
q_x_at_point_dis_gradient = q_array_dis_gradient[:, index_point, 0, :]
q_x_at_point_lin_implicit = q_array_lin_implicit[:, index_point, 0, :]

plt.figure()
plt.plot(t_vec_results_ms, q_x_at_point_reference, \
            label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
for ii in range(n_cases):
    if mask_stable_leapfrog[ii]:
        plt.plot(t_vec_results_ms, q_x_at_point_leapfrog[:, ii], \
                label=fr"LF $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Displacement $q_x$")

plt.figure()
plt.plot(t_vec_results_ms, q_x_at_point_reference, \
            label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
for ii in range(n_cases):
    plt.plot(t_vec_results_ms, q_x_at_point_dis_gradient[:, ii], \
            label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Displacement $q_x$")

plt.figure()
plt.plot(t_vec_results_ms, q_x_at_point_reference, \
            label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
for ii in range(n_cases):
    plt.plot(t_vec_results_ms, q_x_at_point_lin_implicit[:, ii], \
            label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Displacement $q_x$")

q_y_at_point_reference = q_array_reference[:, index_point, 1]
q_y_at_point_leapfrog = q_array_leapfrog[:, index_point, 1, :]
q_y_at_point_dis_gradient = q_array_dis_gradient[:, index_point, 1, :]
q_y_at_point_lin_implicit = q_array_lin_implicit[:, index_point, 1, :]


plt.figure()
plt.plot(t_vec_results_ms, q_y_at_point_reference, \
            label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
for ii in range(n_cases):
    if mask_stable_leapfrog[ii]:
        plt.plot(t_vec_results_ms, q_y_at_point_leapfrog[:, ii], \
                label=fr"LF $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Displacement $q_y$")


plt.figure()
plt.plot(t_vec_results_ms, q_y_at_point_reference, \
            label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
for ii in range(n_cases):
    plt.plot(t_vec_results_ms, q_y_at_point_dis_gradient[:, ii], \
            label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Displacement $q_y$")


plt.figure()
plt.plot(t_vec_results_ms, q_y_at_point_reference, \
            label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
for ii in range(n_cases):
    plt.plot(t_vec_results_ms, q_y_at_point_lin_implicit[:, ii], \
            label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Displacement $q_y$")


# # from matplotlib.transforms import Bbox
# # bbox = Bbox.from_extents(0.75, 0.05, 6.75, 6)  # Adjust these values as needed

# n_times = len(t_vec_output)
# displaced_points_reference = np.zeros((n_times, n_dofs_disp, 3))
# for ii in range(n_times):
#         displaced_points_reference[ii] = q_array_reference[ii] + array_coordinates

# indexes_time= [0, int(n_times/4), int(n_times/2), int(3*n_times/4), n_times-1]

# for index_t in indexes_time:
#      # Create 3D plot
#         fig = plt.figure(figsize=(10, 6))
#         ax = fig.add_subplot(111, projection='3d')

#         # Plot original points in blue
#         ax.scatter(array_coordinates[:, 0], array_coordinates[:, 1], array_coordinates[:, 2], c='blue', label="Reference")

#         # Plot displaced points in red
#         ax.scatter(displaced_points_reference[index_t, :, 0], \
#                    displaced_points_reference[index_t, :, 1], \
#                    displaced_points_reference[index_t, :, 2], c='red', label="Displaced")


plt.show()

