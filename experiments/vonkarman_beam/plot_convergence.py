import matplotlib.pyplot as plt
from src.postprocessing.plot_convergence import plot_convergence
from src.norm_computation import error_norm_time, error_norm_space_time
from parameters import *
import pickle
from src.postprocessing.options import configure_matplotib
configure_matplotib()

norm_type = "L2"
with open(file_results_reference, "rb") as f:
        dict_results_reference = pickle.load(f)

energy_vec_reference = dict_results_reference["energy"]
q_x_array_reference = dict_results_reference["horizontal displacement"]
v_x_array_reference = dict_results_reference["horizontal velocity"]
q_z_array_reference = dict_results_reference["vertical displacement"]
v_z_array_reference = dict_results_reference["vertical velocity"]
    
with open(file_results_leapfrog, "rb") as f:
        dict_results_leapfrog = pickle.load(f)

energy_vec_leapfrog = dict_results_leapfrog["energy"]
q_x_array_leapfrog = dict_results_leapfrog["horizontal displacement"]
v_x_array_leapfrog = dict_results_leapfrog["horizontal velocity"]
q_z_array_leapfrog = dict_results_leapfrog["vertical displacement"]
v_z_array_leapfrog = dict_results_leapfrog["vertical velocity"]
comp_time_leapfrog = dict_results_leapfrog["elapsed time"]

avg_diff_E_leapfrog = np.mean(np.diff(energy_vec_leapfrog, axis=0))

with open(file_results_dis_gradient, "rb") as f:
        dict_results_dis_gradient = pickle.load(f)

energy_vec_dis_gradient = dict_results_dis_gradient["energy"]
q_x_array_dis_gradient = dict_results_dis_gradient["horizontal displacement"]
v_x_array_dis_gradient = dict_results_dis_gradient["horizontal velocity"]
q_z_array_dis_gradient = dict_results_dis_gradient["vertical displacement"]
v_z_array_dis_gradient = dict_results_dis_gradient["vertical velocity"]
comp_time_dis_gradient = dict_results_dis_gradient["elapsed time"]

avg_diff_E_dis_gradient = np.mean(np.diff(energy_vec_dis_gradient, axis=0))

with open(file_results_lin_implicit, "rb") as f:
        dict_results_lin_implicit = pickle.load(f)

energy_vec_lin_implicit = dict_results_lin_implicit["energy"]
q_x_array_lin_implicit = dict_results_lin_implicit["horizontal displacement"]
v_x_array_lin_implicit = dict_results_lin_implicit["horizontal velocity"]
q_z_array_lin_implicit = dict_results_lin_implicit["vertical displacement"]
v_z_array_lin_implicit = dict_results_lin_implicit["vertical velocity"]
comp_time_lin_implicit = dict_results_lin_implicit["elapsed time"]

avg_diff_E_lin_implicit = np.mean(np.diff(energy_vec_lin_implicit, axis=0))


error_q_x_leapfrog = np.zeros(n_cases)
error_q_z_leapfrog = np.zeros(n_cases)
error_v_x_leapfrog = np.zeros(n_cases)
error_v_z_leapfrog = np.zeros(n_cases)

error_q_x_dis_gradient = np.zeros(n_cases)
error_q_z_dis_gradient = np.zeros(n_cases)
error_v_x_dis_gradient = np.zeros(n_cases)
error_v_z_dis_gradient = np.zeros(n_cases)

error_q_x_lin_implicit = np.zeros(n_cases)
error_q_z_lin_implicit = np.zeros(n_cases)
error_v_x_lin_implicit = np.zeros(n_cases)
error_v_z_lin_implicit = np.zeros(n_cases)

 # # Compute error

for ii in range(n_cases):
        error_q_x_leapfrog[ii] = error_norm_space_time(q_x_array_reference, \
                                                           q_x_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        error_q_z_leapfrog[ii] = error_norm_space_time(q_z_array_reference, \
                                                           q_z_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        error_v_x_leapfrog[ii] = error_norm_space_time(v_x_array_reference, \
                                                           v_x_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        error_v_z_leapfrog[ii] = error_norm_space_time(v_z_array_reference, \
                                                           v_z_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        
        error_q_x_dis_gradient[ii] = error_norm_space_time(q_x_array_reference, \
                                                           q_x_array_dis_gradient[:, :, ii], dt_output, norm=norm_type)
        error_q_z_dis_gradient[ii] = error_norm_space_time(q_z_array_reference, \
                                                           q_z_array_dis_gradient[:, :, ii], dt_output, norm=norm_type)
        error_v_x_dis_gradient[ii] = error_norm_space_time(v_x_array_reference, \
                                                           v_x_array_dis_gradient[:, :, ii], dt_output, norm=norm_type)
        error_v_z_dis_gradient[ii] = error_norm_space_time(v_z_array_reference, \
                                                           v_z_array_dis_gradient[:, :, ii], dt_output, norm=norm_type)

        error_q_x_lin_implicit[ii] = error_norm_space_time(q_x_array_reference, \
                                                           q_x_array_lin_implicit[:, :, ii], dt_output, norm=norm_type)
        error_q_z_lin_implicit[ii] = error_norm_space_time(q_z_array_reference, \
                                                           q_z_array_lin_implicit[:, :, ii], dt_output, norm=norm_type)
        error_v_x_lin_implicit[ii] = error_norm_space_time(v_x_array_reference, \
                                                           v_x_array_lin_implicit[:, :, ii], dt_output, norm=norm_type)
        error_v_z_lin_implicit[ii] = error_norm_space_time(v_z_array_reference, \
                                                           v_z_array_lin_implicit[:, :, ii], dt_output, norm=norm_type)


dict_hor_position = {"Linear implicit": error_q_x_lin_implicit, \
                "Discrete gradient": error_q_x_dis_gradient, \
                "Leapfrog": error_q_x_leapfrog}

dict_ver_position = {"Linear implicit": error_q_z_lin_implicit, \
                "Discrete gradient": error_q_z_dis_gradient, \
                "Leapfrog": error_q_z_leapfrog}

dict_hor_velocity = {"Linear implicit": error_v_x_lin_implicit, \
                "Discrete gradient": error_v_x_dis_gradient, \
                "Leapfrog": error_v_x_leapfrog}

dict_ver_velocity = {"Linear implicit": error_v_z_lin_implicit, \
                     "Discrete gradient": error_v_z_dis_gradient, 
                     "Leapfrog": error_v_z_leapfrog}

str_xlabel = '$\log \Delta t \; \mathrm{[s]}$'
plot_convergence(time_step_vec, dict_hor_position, xlabel=str_xlabel, ylabel="$\log \Delta q_x$", \
                title='Error position $q_x$', savefig=f"{directory_results}convergence_horizontal_position.pdf")
plot_convergence(time_step_vec, dict_hor_velocity, xlabel=str_xlabel, ylabel="$\log \Delta v_x$",  \
                 title='Error velocity $v_x$', savefig=f"{directory_results}convergence_horizontal_velocity.pdf")

plot_convergence(time_step_vec, dict_ver_position, xlabel=str_xlabel, ylabel="$\log \Delta q$", \
                title='Error position $q_z$', savefig=f"{directory_results}convergence_vertical_position.pdf")
plot_convergence(time_step_vec, dict_ver_velocity, xlabel=str_xlabel, ylabel="$\log \Delta v$",  \
                 title='Error velocity $v_z$', savefig=f"{directory_results}convergence_vertical_velocity.pdf")



plt.figure()
plt.loglog(time_step_vec, comp_time_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, comp_time_lin_implicit, '+-', label='Linear implicit')
plt.loglog(time_step_vec, comp_time_leapfrog, '+-', label='Leapfrog')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel("$\log \\tau$")
plt.legend()
plt.grid(True)
plt.title("Computational time [s]")
plt.savefig(f"{directory_results}computational_time.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.show()
