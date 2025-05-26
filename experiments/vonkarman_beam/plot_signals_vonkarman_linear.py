import matplotlib.pyplot as plt
from experiments.vonkarman_beam.parameters_vonkarman import *
import pickle
import matplotlib.patches as mpatches
from matplotlib import cm
from src.postprocessing.options import configure_matplotib
configure_matplotib()

fraction_total = 1
n_plot_convergence = int(max_output*fraction_total)


dt_linear = dt_base/2**3
with open(file_time, "rb") as f:
        dict_time = pickle.load(f)

t_vec_results_ms = 1e3*dict_time["Time"][:n_plot_convergence]

with open(file_results_linear, "rb") as f:
        dict_results_linear = pickle.load(f)

energy_vec_linear = dict_results_linear["energy"][:n_plot_convergence]
q_x_array_linear = dict_results_linear["horizontal displacement"][:n_plot_convergence, :]
v_x_array_linear = dict_results_linear["horizontal velocity"][:n_plot_convergence, :]
q_z_array_linear = dict_results_linear["vertical displacement"][:n_plot_convergence, :]
v_z_array_linear = dict_results_linear["vertical velocity"][:n_plot_convergence, :]
    
hor_disp_at_point_linear = q_x_array_linear[:, index_point]
ver_disp_at_point_linear = q_z_array_linear[:, index_point]

plt.figure()
plt.plot(t_vec_results_ms, hor_disp_at_point_linear, \
            label=fr"Ref $\Delta t = {dt_linear*1e6:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Horizontal displacement")

ver_disp_at_point_linear = q_z_array_linear[:, index_point]



plt.figure()
plt.plot(t_vec_results_ms, ver_disp_at_point_linear, \
            label=fr"Ref $\Delta t = {dt_linear*1e6:.1f} \; \mathrm{{[\mu s]}}$")

plt.legend()
plt.xlabel("Time [ms]")
plt.title("Vertical displacement")


from matplotlib.transforms import Bbox
bbox = Bbox.from_extents(0.75, 0.05, 6.75, 6)  # Adjust these values as needed

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
plt.savefig(f"{directory_images}horizontal_displacement_vonkarman.pdf",dpi='figure',\
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
plt.savefig(f"{directory_images}vertical_displacement_vonkarman.pdf", dpi='figure', \
            format='pdf', bbox_inches=bbox)
plt.show()


