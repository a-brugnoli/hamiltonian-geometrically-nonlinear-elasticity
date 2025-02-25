from experiments.vonkarman_beam.parameters_vonkarman import *
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from src.postprocessing.options import configure_matplotib
configure_matplotib()


dt = time_step_vec[0]
beam.set_time_step(dt)

t0_lin_implicit_static_cond = time.perf_counter()
dict_results_lin_implicit_static_cond = beam.linear_implicit_static_condensation(save_vars=True)
tf_lin_implicit_static_cond = time.perf_counter()

q_x_array_lin_implicit_static_cond = dict_results_lin_implicit_static_cond["horizontal displacement"]
v_x_array_lin_implicit_static_cond = dict_results_lin_implicit_static_cond["horizontal velocity"]
q_z_array_lin_implicit_static_cond = dict_results_lin_implicit_static_cond["vertical displacement"]
v_z_array_lin_implicit_static_cond = dict_results_lin_implicit_static_cond["vertical velocity"]

hor_disp_at_point_lin_implicit_static_cond = q_x_array_lin_implicit_static_cond[:, index_point]
ver_disp_at_point_lin_implicit_static_cond = q_z_array_lin_implicit_static_cond[:, index_point]

t0_lin_implicit = time.perf_counter()
dict_results_lin_implicit = beam.linear_implicit(save_vars=True)
tf_lin_implicit = time.perf_counter()

q_x_array_lin_implicit = dict_results_lin_implicit["horizontal displacement"]
v_x_array_lin_implicit = dict_results_lin_implicit["horizontal velocity"]
q_z_array_lin_implicit = dict_results_lin_implicit["vertical displacement"]
v_z_array_lin_implicit = dict_results_lin_implicit["vertical velocity"]

hor_disp_at_point_lin_implicit = q_x_array_lin_implicit[:, index_point]
ver_disp_at_point_lin_implicit = q_z_array_lin_implicit[:, index_point]


elapsed_vec_lin_implicit_static_cond = tf_lin_implicit_static_cond - t0_lin_implicit_static_cond
elapsed_vec_lin_implicit = tf_lin_implicit - t0_lin_implicit
print(f"Elapsed time Linear implicit static condensation : {elapsed_vec_lin_implicit_static_cond}")
print(f"Elapsed time Linear implicit : {elapsed_vec_lin_implicit}")

plt.figure()
plt.plot(t_vec_output_ms, ver_disp_at_point_lin_implicit, \
            label=fr"Ref $\Delta t = {dt*1e6:.1f} \; \mathrm{{[\mu s]}}$")
plt.plot(t_vec_output_ms, ver_disp_at_point_lin_implicit_static_cond, \
            label=fr"DG $\Delta t = {dt*1e6:.1f} \; \mathrm{{[\mu s]}}$")
plt.legend()
plt.xlabel("Time [ms]")
plt.title("Vertical displacement")

diff_q_x_array = q_x_array_lin_implicit_static_cond - q_x_array_lin_implicit
diff_q_z_array = q_z_array_lin_implicit_static_cond - q_z_array_lin_implicit

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_array, T_array_ms = np.meshgrid(x_vec, t_vec_output_ms)
surf = ax.plot_surface(X_array, T_array_ms, diff_q_x_array, \
                        cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
# Add labels and legend
ax.set_xlabel("$x \; \mathrm{[m]}$")
ax.set_ylabel("$t \; \mathrm{[ms]}$")
ax.set_zlabel("$\Delta q_x \; \mathrm{[m]}$ ")
ax.set_title("Difference Horizontal displacement")
# Set viewing angle
ax.view_init(elev=30, azim=45)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_array, T_array_ms = np.meshgrid(x_vec, t_vec_output_ms)
surf = ax.plot_surface(X_array, T_array_ms, diff_q_z_array, \
                       cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=True)
# Add labels and legend
ax.set_xlabel("$x \; \mathrm{[m]}$")
ax.set_ylabel("$t \; \mathrm{[ms]}$")
ax.set_zlabel("$\Delta q_z \; \mathrm{[mm]}$ ")
ax.set_title("Differnce Vertical displacement")
# Set viewing angle
ax.view_init(elev=30, azim=45)
plt.tight_layout()

plt.show()