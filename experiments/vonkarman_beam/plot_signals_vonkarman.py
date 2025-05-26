import matplotlib.pyplot as plt
from experiments.vonkarman_beam.parameters_vonkarman import *
import pickle
import matplotlib.patches as mpatches
from matplotlib import cm
from src.postprocessing.options import configure_matplotib
configure_matplotib()

fraction_total = 1
n_plot_convergence = int(max_output*fraction_total)

with open(file_time, "rb") as f:
        dict_time = pickle.load(f)

t_vec_results_ms = 1e3*dict_time["Time"][:n_plot_convergence]

with open(file_results_reference, "rb") as f:
        dict_results_reference = pickle.load(f)

energy_vec_reference = dict_results_reference["energy"][:n_plot_convergence]
q_x_array_reference = dict_results_reference["horizontal displacement"][:n_plot_convergence, :]
v_x_array_reference = dict_results_reference["horizontal velocity"][:n_plot_convergence, :]
q_z_array_reference = dict_results_reference["vertical displacement"][:n_plot_convergence, :]
v_z_array_reference = dict_results_reference["vertical velocity"][:n_plot_convergence, :]
    
with open(file_results_linear, "rb") as f:
        dict_results_linear = pickle.load(f)

energy_vec_linear = dict_results_linear["energy"][:n_plot_convergence]
q_x_array_linear = dict_results_linear["horizontal displacement"][:n_plot_convergence, :]
v_x_array_linear = dict_results_linear["horizontal velocity"][:n_plot_convergence, :]
q_z_array_linear = dict_results_linear["vertical displacement"][:n_plot_convergence, :]
v_z_array_linear = dict_results_linear["vertical velocity"][:n_plot_convergence, :]
    
hor_disp_at_point_linear = q_x_array_linear[:, index_point]
ver_disp_at_point_linear = q_z_array_linear[:, index_point]


with open(file_results_leapfrog, "rb") as f:
        dict_results_leapfrog = pickle.load(f)

energy_vec_leapfrog = dict_results_leapfrog["energy"][:n_plot_convergence]
q_x_array_leapfrog = dict_results_leapfrog["horizontal displacement"][:n_plot_convergence, :]
v_x_array_leapfrog = dict_results_leapfrog["horizontal velocity"][:n_plot_convergence, :]
q_z_array_leapfrog = dict_results_leapfrog["vertical displacement"][:n_plot_convergence, :]
v_z_array_leapfrog = dict_results_leapfrog["vertical velocity"][:n_plot_convergence, :]
comp_time_leapfrog = dict_results_leapfrog["elapsed time"]

diff_E_leapfrog = np.diff(energy_vec_leapfrog, axis=0)


with open(file_results_dis_gradient, "rb") as f:
        dict_results_dis_gradient = pickle.load(f)

energy_vec_dis_gradient = dict_results_dis_gradient["energy"][:n_plot_convergence]
q_x_array_dis_gradient = dict_results_dis_gradient["horizontal displacement"][:n_plot_convergence, :]
v_x_array_dis_gradient = dict_results_dis_gradient["horizontal velocity"][:n_plot_convergence, :]
q_z_array_dis_gradient = dict_results_dis_gradient["vertical displacement"][:n_plot_convergence, :]
v_z_array_dis_gradient = dict_results_dis_gradient["vertical velocity"][:n_plot_convergence, :]
comp_time_dis_gradient = dict_results_dis_gradient["elapsed time"]

diff_E_dis_gradient = np.diff(energy_vec_dis_gradient, axis=0)

with open(file_results_lin_implicit, "rb") as f:
        dict_results_lin_implicit = pickle.load(f)

energy_vec_lin_implicit = dict_results_lin_implicit["energy"][:n_plot_convergence]
q_x_array_lin_implicit = dict_results_lin_implicit["horizontal displacement"][:n_plot_convergence, :]
v_x_array_lin_implicit = dict_results_lin_implicit["horizontal velocity"][:n_plot_convergence, :]
q_z_array_lin_implicit = dict_results_lin_implicit["vertical displacement"][:n_plot_convergence, :]
v_z_array_lin_implicit = dict_results_lin_implicit["vertical velocity"][:n_plot_convergence, :]
comp_time_lin_implicit = dict_results_lin_implicit["elapsed time"]

diff_E_lin_implicit = np.diff(energy_vec_lin_implicit, axis=0)

# plt.figure()
# for ii in range(n_cases):
#     plt.plot(t_vec_results_ms, energy_vec_dis_gradient[:, ii], '-.', \
#              label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
#     plt.plot(t_vec_results_ms, energy_vec_lin_implicit[:, ii], ':', \
#              label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
#     if mask_stable_leapfrog[ii]:
#         plt.plot(t_vec_results_ms, energy_vec_leapfrog[:, ii], '-.', \
#              label=fr"LF $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
# plt.legend()
# plt.xlabel("Time [ms]")
# plt.title("Energy")


# plt.figure()
# for ii in range(n_cases):
#     plt.plot(t_vec_results_ms[1:], diff_E_dis_gradient[:, ii], '-.', \
#             label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
#     plt.plot(t_vec_results_ms[1:], diff_E_lin_implicit[:, ii], ':', \
#             label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
#     if mask_stable_leapfrog[ii]:
#         plt.plot(t_vec_results_ms[1:], diff_E_leapfrog[:, ii], '-.', \
#                 label=fr"LF $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
# plt.legend()
# plt.xlabel("Time [ms]")
# plt.ylabel("$H(t_{n+1}) - H(t_n)$")
# plt.title("Difference Energy")

hor_disp_at_point_reference = q_x_array_reference[:, index_point]
hor_disp_at_point_leapfrog = q_x_array_leapfrog[:, index_point]
hor_disp_at_point_dis_gradient = q_x_array_dis_gradient[:, index_point]
hor_disp_at_point_lin_implicit = q_x_array_lin_implicit[:, index_point]

plt.figure()
plt.plot(t_vec_results_ms, hor_disp_at_point_reference, \
            label=fr"Nonlinear")
plt.plot(t_vec_results_ms, hor_disp_at_point_linear, \
            label=fr"Linear")
# for ii in range(n_cases):
#     if mask_stable_leapfrog[ii]:
#         plt.plot(t_vec_results_ms, hor_disp_at_point_leapfrog[:, ii], \
#                 label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
#     plt.plot(t_vec_results_ms, hor_disp_at_point_dis_gradient[:, ii], \
#              label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
#     plt.plot(t_vec_results_ms, hor_disp_at_point_lin_implicit[:, ii], \
#              label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Horizontal displacement")
plt.savefig(f"{directory_images}horizontal_displacement_at_point_vonkarman.pdf",dpi='figure',\
             format='pdf')

ver_disp_at_point_reference = q_z_array_reference[:, index_point]
ver_disp_at_point_leapfrog = q_z_array_leapfrog[:, index_point]
ver_disp_at_point_dis_gradient = q_z_array_dis_gradient[:, index_point]
ver_disp_at_point_lin_implicit = q_z_array_lin_implicit[:, index_point]


plt.figure()
plt.plot(t_vec_results_ms, ver_disp_at_point_reference, \
            label=fr"Nonlinear")
plt.plot(t_vec_results_ms, ver_disp_at_point_linear, \
            label=fr"Linear")
# for ii in range(n_cases):
#     if mask_stable_leapfrog[ii]:
#         plt.plot(t_vec_results_ms, ver_disp_at_point_leapfrog[:, ii], \
#                 label=fr"LF $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Vertical displacement")
plt.savefig(f"{directory_images}vertical_displacement_at_point_vonkarman.pdf",dpi='figure',\
             format='pdf')

# plt.figure()
# plt.plot(t_vec_results_ms, ver_disp_at_point_reference, \
#             label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
# for ii in range(n_cases):
#     plt.plot(t_vec_results_ms, ver_disp_at_point_dis_gradient[:, ii], \
#             label=fr"DG $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
# plt.legend()
# plt.xlabel("Time [ms]")
# plt.title("Vertical displacement")


# plt.figure()
# plt.plot(t_vec_results_ms, ver_disp_at_point_reference, \
#             label=fr"Ref $\Delta t = {dt_reference*1e6:.1f} \; \mathrm{{[\mu s]}}$")
# for ii in range(n_cases):
#     plt.plot(t_vec_results_ms, ver_disp_at_point_lin_implicit[:, ii], \
#             label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
# plt.legend()
# plt.xlabel("Time [ms]")
# plt.title("Vertical displacement")


from matplotlib.transforms import Bbox
bbox = Bbox.from_extents(0.75, 0.05, 6.75, 6)  # Adjust these values as needed

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_array, T_array_ms = np.meshgrid(x_vec, t_vec_results_ms)
surf = ax.plot_surface(X_array, T_array_ms, 1e6*q_x_array_reference, \
                        cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
# Add labels and legend
ax.set_xlabel("$x \; \mathrm{[m]}$")
ax.set_ylabel("$t \; \mathrm{[ms]}$")
ax.set_zlabel("$q_x \; [\mu \mathrm{m}]$ ")
ax.set_title("Horizontal displacement")
# Set viewing angle
ax.view_init(elev=30, azim=45)
plt.savefig(f"{directory_images}horizontal_displacement_vonkarman.pdf",dpi='figure',\
             format='pdf', bbox_inches=bbox)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_array, T_array_ms = np.meshgrid(x_vec, t_vec_results_ms)
surf = ax.plot_surface(X_array, T_array_ms, 1e3*q_z_array_reference, \
                       cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
# Add labels and legend
ax.set_xlabel("$x \; \mathrm{[m]}$")
ax.set_ylabel("$t \; \mathrm{[ms]}$")
ax.set_zlabel("$q_z \; \mathrm{[mm]}$ ")
ax.set_title("Vertical displacement")
# Set viewing angle
ax.view_init(elev=30, azim=45)
# ax.set_box_aspect(None, zoom=zoom)
plt.savefig(f"{directory_images}vertical_displacement_vonkarman.pdf", dpi='figure', \
            format='pdf', bbox_inches=bbox)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_array, T_array_ms = np.meshgrid(x_vec, t_vec_results_ms)
surf = ax.plot_surface(X_array, T_array_ms, 1e6*q_x_array_linear, \
                        cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
# Add labels and legend
ax.set_xlabel("$x \; \mathrm{[m]}$")
ax.set_ylabel("$t \; \mathrm{[ms]}$")
ax.set_zlabel("$q_x \; [\mu \mathrm{m}]$ ")
ax.set_title("Horizontal displacement")
# Set viewing angle
ax.view_init(elev=30, azim=45)
plt.savefig(f"{directory_images}horizontal_displacement_vonkarman_linear.pdf",dpi='figure',\
             format='pdf', bbox_inches=bbox)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_array, T_array_ms = np.meshgrid(x_vec, t_vec_results_ms)
surf = ax.plot_surface(X_array, T_array_ms, 1e3*q_z_array_linear, \
                       cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
# Add labels and legend
ax.set_xlabel("$x \; \mathrm{[m]}$")
ax.set_ylabel("$t \; \mathrm{[ms]}$")
ax.set_zlabel("$q_z \; \mathrm{[mm]}$ ")
ax.set_title("Vertical displacement")
# Set viewing angle
ax.view_init(elev=30, azim=45)
# ax.set_box_aspect(None, zoom=zoom)
plt.savefig(f"{directory_images}vertical_displacement_vonkarman_linear.pdf", dpi='figure', \
            format='pdf', bbox_inches=bbox)
plt.show()



# diff_q_z_array_lin_implicit = q_z_array_reference[:,:,np.newaxis] - q_z_array_lin_implicit

# fig_surfaces = plt.figure()
# ax_surfaces = fig_surfaces.add_subplot(111, projection='3d')

# colormaps = [cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.cividis, cm.turbo]
# proxy_artists = []
# for ii in range(n_cases):
#     # Choose colormap for this surface
#     current_cmap = colormaps[ii % len(colormaps)]
    
#     # Create surface with alpha transparency
#     surf = ax_surfaces.plot_surface(X_array, T_array_ms, diff_q_z_array_lin_implicit[:,:, ii], \
#                                     cmap=current_cmap, alpha=0.7, linewidth=0, antialiased=True)
    
#     # Create proxy artist for legend
#     proxy = mpatches.Patch(color=current_cmap(0.7), 
#                            label=fr"LI $\Delta t = {time_step_vec_mus[ii]:.1f} \; \mathrm{{[\mu s]}}$")
#     proxy_artists.append(proxy)

# # Add labels and legend
# ax_surfaces.set_xlabel("$x \; \mathrm{[m]}$")
# ax_surfaces.set_ylabel("$t \; \mathrm{[ms]}$")
# ax_surfaces.set_zlabel("$q_z \; \mathrm{[m]}$ ")
# ax_surfaces.set_title("Vertical displacement LI")
# ax_surfaces.legend(handles=proxy_artists, loc='upper right')
# # Set viewing angle
# ax_surfaces.view_init(elev=30, azim=45)
# plt.tight_layout()


# plt.show()

