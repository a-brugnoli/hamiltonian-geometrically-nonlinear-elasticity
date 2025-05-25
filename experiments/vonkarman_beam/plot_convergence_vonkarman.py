import matplotlib.pyplot as plt
from src.postprocessing.plot_convergence import plot_convergence
from src.norm_computation import error_norm_time, error_norm_space_time
from experiments.vonkarman_beam.parameters_vonkarman import *
import pickle
from src.postprocessing.options import configure_matplotib
configure_matplotib()

norm_type = "L2"
with open(file_results_reference, "rb") as f:
        dict_results_reference = pickle.load(f)

fraction_total = 100
n_plot_convergence = int(max_output/fraction_total) 
energy_vec_reference = dict_results_reference["energy"][:n_plot_convergence]
q_x_array_reference = dict_results_reference["horizontal displacement"][:n_plot_convergence, :]
v_x_array_reference = dict_results_reference["horizontal velocity"][:n_plot_convergence, :]
q_z_array_reference = dict_results_reference["vertical displacement"][:n_plot_convergence, :]
v_z_array_reference = dict_results_reference["vertical velocity"][:n_plot_convergence, :]
    
with open(file_results_leapfrog, "rb") as f:
        dict_results_leapfrog = pickle.load(f)

comp_time_leapfrog = dict_results_leapfrog["elapsed time"]

energy_vec_leapfrog = dict_results_leapfrog["energy"][:n_plot_convergence]

q_x_array_leapfrog = dict_results_leapfrog["horizontal displacement"][:n_plot_convergence, :]
v_x_array_leapfrog = dict_results_leapfrog["horizontal velocity"][:n_plot_convergence, :]
q_z_array_leapfrog = dict_results_leapfrog["vertical displacement"][:n_plot_convergence, :]
v_z_array_leapfrog = dict_results_leapfrog["vertical velocity"][:n_plot_convergence, :]

diff_energy_vec_leapfrog = np.abs(np.diff(energy_vec_leapfrog, axis=0))
avg_diff_E_leapfrog = np.mean(diff_energy_vec_leapfrog, axis=0)

with open(file_results_dis_gradient, "rb") as f:
        dict_results_dis_gradient = pickle.load(f)

comp_time_dis_gradient = dict_results_dis_gradient["elapsed time"]

energy_vec_dis_gradient = dict_results_dis_gradient["energy"][:n_plot_convergence]

q_x_array_dis_gradient = dict_results_dis_gradient["horizontal displacement"][:n_plot_convergence, :]
v_x_array_dis_gradient = dict_results_dis_gradient["horizontal velocity"][:n_plot_convergence, :]
q_z_array_dis_gradient = dict_results_dis_gradient["vertical displacement"][:n_plot_convergence, :]
v_z_array_dis_gradient = dict_results_dis_gradient["vertical velocity"][:n_plot_convergence, :]

diff_energy_dis_gradient = np.abs(np.diff(energy_vec_dis_gradient, axis=0))
avg_diff_E_dis_gradient = np.mean(diff_energy_dis_gradient, axis=0)

with open(file_results_lin_implicit, "rb") as f:
        dict_results_lin_implicit = pickle.load(f)

comp_time_lin_implicit = dict_results_lin_implicit["elapsed time"]

energy_vec_lin_implicit = dict_results_lin_implicit["energy"][:n_plot_convergence]

q_x_array_lin_implicit = dict_results_lin_implicit["horizontal displacement"][:n_plot_convergence, :]
v_x_array_lin_implicit = dict_results_lin_implicit["horizontal velocity"][:n_plot_convergence, :]
q_z_array_lin_implicit = dict_results_lin_implicit["vertical displacement"][:n_plot_convergence, :]
v_z_array_lin_implicit = dict_results_lin_implicit["vertical velocity"][:n_plot_convergence, :]

diff_energy_vec_lin_implicit = np.abs(np.diff(energy_vec_lin_implicit, axis=0))
avg_diff_E_lin_implicit = np.mean(diff_energy_vec_lin_implicit, axis=0)

error_q_x_dis_gradient = np.zeros(n_cases)
error_q_z_dis_gradient = np.zeros(n_cases)
error_v_x_dis_gradient = np.zeros(n_cases)
error_v_z_dis_gradient = np.zeros(n_cases)

error_q_x_lin_implicit = np.zeros(n_cases)
error_q_z_lin_implicit = np.zeros(n_cases)
error_v_x_lin_implicit = np.zeros(n_cases)
error_v_z_lin_implicit = np.zeros(n_cases)

 # # Compute error
error_q_x_leapfrog = np.zeros(n_cases)
error_q_z_leapfrog = np.zeros(n_cases)
error_v_x_leapfrog = np.zeros(n_cases)
error_v_z_leapfrog = np.zeros(n_cases) 

# error_q_x_leapfrog = np.zeros(n_cases_stable_leapfrog)
# error_q_z_leapfrog = np.zeros(n_cases_stable_leapfrog)
# error_v_x_leapfrog = np.zeros(n_cases_stable_leapfrog)
# error_v_z_leapfrog = np.zeros(n_cases_stable_leapfrog)
# kk=0
for ii in range(n_cases):
        error_q_x_leapfrog[ii] = error_norm_space_time(q_x_array_reference, \
                                                q_x_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        error_q_z_leapfrog[ii] = error_norm_space_time(q_z_array_reference, \
                                                q_z_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        error_v_x_leapfrog[ii] = error_norm_space_time(v_x_array_reference, \
                                                v_x_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        error_v_z_leapfrog[ii] = error_norm_space_time(v_z_array_reference, \
                                                v_z_array_leapfrog[:, :, ii], dt_output, norm=norm_type)

        # if mask_stable_leapfrog[ii]:
        #         error_q_x_leapfrog[kk] = error_norm_space_time(q_x_array_reference, \
        #                                                 q_x_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        #         error_q_z_leapfrog[kk] = error_norm_space_time(q_z_array_reference, \
        #                                                 q_z_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        #         error_v_x_leapfrog[kk] = error_norm_space_time(v_x_array_reference, \
        #                                                 v_x_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        #         error_v_z_leapfrog[kk] = error_norm_space_time(v_z_array_reference, \
        #                                                 v_z_array_leapfrog[:, :, ii], dt_output, norm=norm_type)
        #         kk+=1
        
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



dict_hor_displacement = {"Discrete gradient": error_q_x_dis_gradient, \
                         "Linear implicit": error_q_x_lin_implicit, \
                        "Leapfrog": error_q_x_leapfrog}

dict_ver_displacement = {"Discrete gradient": error_q_z_dis_gradient, \
                        "Linear implicit": error_q_z_lin_implicit, \
                        "Leapfrog": error_q_z_leapfrog}

dict_hor_velocity = {"Discrete gradient": error_v_x_dis_gradient, \
                "Linear implicit": error_v_x_lin_implicit, \
                "Leapfrog": error_v_x_leapfrog}

dict_ver_velocity = {"Discrete gradient": error_v_z_dis_gradient,  
                     "Linear implicit": error_v_z_lin_implicit, 
                     "Leapfrog": error_v_z_leapfrog}

str_xlabel = '$\log \Delta t \; \mathrm{[s]}$'
plot_convergence(time_step_vec, dict_hor_displacement, rate=True, xlabel=str_xlabel, ylabel="$\log \epsilon_{q_x}$", \
                title='Error displacement $q_x$', savefig=f"{directory_images}convergence_horizontal_displacement_vonkarman")

plot_convergence(time_step_vec, dict_hor_velocity, rate=True, xlabel=str_xlabel, ylabel="$\log \epsilon_{v_x}$",  \
                 title='Error velocity $v_x$', savefig=f"{directory_images}convergence_horizontal_velocity_vonkarman")

plot_convergence(time_step_vec, dict_ver_displacement, rate=True, xlabel=str_xlabel, ylabel="$\log \epsilon_{q_z}$", \
                title='Error displacement $q_z$', savefig=f"{directory_images}convergence_vertical_displacement_vonkarman")

plot_convergence(time_step_vec, dict_ver_velocity, rate=True, xlabel=str_xlabel, ylabel="$\log \epsilon_{v_z}$",  \
                 title='Error velocity $v_z$', savefig=f"{directory_images}convergence_vertical_velocity_vonkarman")



plt.figure()
plt.loglog(time_step_vec, comp_time_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, comp_time_lin_implicit, '+--', label='Linear implicit')
plt.loglog(time_step_vec, comp_time_leapfrog, '^:', label='Leapfrog')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel(r"$\log T_{\rm comp}$")
plt.legend()
plt.grid(True)
plt.title("Computational time [s]")
plt.savefig(f"{directory_images}computational_time_vonkarman.pdf", dpi='figure', format='pdf', bbox_inches="tight")


plt.figure()
plt.loglog(time_step_vec, avg_diff_E_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, avg_diff_E_lin_implicit, '+--', label='Linear implicit')
plt.loglog(time_step_vec, avg_diff_E_leapfrog, '^:', label='Leapfrog')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel(r"$\frac{1}{N_t}\sum_{n=0}^{N_t}|H_{n+1} - H_{n}|$")
plt.legend()
plt.grid(True)
plt.title("Mean of energy difference")
plt.savefig(f"{directory_images}energy_difference_vonkarman.pdf", dpi='figure', format='pdf', bbox_inches="tight")


plt.show()
