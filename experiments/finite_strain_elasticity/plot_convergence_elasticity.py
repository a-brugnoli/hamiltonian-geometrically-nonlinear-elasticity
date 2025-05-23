import matplotlib.pyplot as plt
from src.postprocessing.plot_convergence import plot_convergence
from src.norm_computation import error_norm_time, error_norm_space_time
from experiments.finite_strain_elasticity.parameters_elasticity import *
import pickle
from src.postprocessing.options import configure_matplotib
configure_matplotib()

norm_type = "L2"
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

diff_energy_vec_leapfrog = np.abs(np.diff(energy_vec_leapfrog, axis=0))
avg_diff_E_leapfrog = np.mean(diff_energy_vec_leapfrog, axis=0)

with open(file_results_dis_gradient, "rb") as f:
        dict_results_dis_gradient = pickle.load(f)

energy_vec_dis_gradient = dict_results_dis_gradient["energy"]
q_array_dis_gradient = dict_results_dis_gradient["displacement"]
v_array_dis_gradient = dict_results_dis_gradient["velocity"]
comp_time_dis_gradient = dict_results_dis_gradient["elapsed time"]

diff_energy_dis_gradient = np.abs(np.diff(energy_vec_dis_gradient, axis=0))
avg_diff_E_dis_gradient = np.mean(diff_energy_dis_gradient, axis=0)

with open(file_results_lin_implicit, "rb") as f:
        dict_results_lin_implicit = pickle.load(f)

energy_vec_lin_implicit = dict_results_lin_implicit["energy"]
q_array_lin_implicit = dict_results_lin_implicit["displacement"]
v_array_lin_implicit = dict_results_lin_implicit["velocity"]
comp_time_lin_implicit = dict_results_lin_implicit["elapsed time"]

diff_energy_vec_lin_implicit = np.abs(np.diff(energy_vec_lin_implicit, axis=0))
avg_diff_E_lin_implicit = np.mean(diff_energy_vec_lin_implicit, axis=0)

with open(file_results_semiexplicit, "rb") as f:
        dict_results_semiexplicit = pickle.load(f)

if True:
        energy_vec_semiexplicit = dict_results_semiexplicit["energy"]
        q_array_semiexplicit = dict_results_semiexplicit["displacement"]
        v_array_semiexplicit = dict_results_semiexplicit["velocity"]
        comp_time_semiexplicit = dict_results_semiexplicit["elapsed time"]

        diff_energy_vec_semiexplicit = np.abs(np.diff(energy_vec_semiexplicit, axis=0))
        avg_diff_E_semiexplicit = np.mean(diff_energy_vec_semiexplicit, axis=0)

error_q_dis_gradient = np.zeros(n_cases)
error_v_dis_gradient = np.zeros(n_cases)

error_q_lin_implicit = np.zeros(n_cases)
error_v_lin_implicit = np.zeros(n_cases)

 # # Compute error
error_q_leapfrog = np.zeros(n_cases)
error_v_leapfrog = np.zeros(n_cases) 

error_q_semiexplicit = np.zeros(n_cases)
error_v_semiexplicit = np.zeros(n_cases)

# error_q_leapfrog = np.zeros(n_cases_stable_leapfrog)
# error_v_leapfrog = np.zeros(n_cases_stable_leapfrog)
# kk=0
for ii in range(n_cases):
        error_q_leapfrog[ii] += error_norm_space_time(q_array_reference, \
                                                q_array_leapfrog[:, :, :, ii], dt_output, norm=norm_type)
        error_v_leapfrog[ii] += error_norm_space_time(v_array_reference, \
                                                v_array_leapfrog[:, :, :, ii], dt_output, norm=norm_type)

# if mask_stable_leapfrog[ii]:
#         error_q_leapfrog[kk] = error_norm_space_time(q_array_reference, \
#                                                 q_array_leapfrog[:, :, :, ii], dt_output, norm=norm_type)
#         error_v_leapfrog[kk] = error_norm_space_time(v_array_reference, \
#                                                 v_array_leapfrog[:, :, :, ii], dt_output, norm=norm_type)
#         kk+=1

        error_q_dis_gradient[ii] = error_norm_space_time(q_array_reference, \
                                                        q_array_dis_gradient[:, :, :, ii], dt_output, norm=norm_type)
        error_v_dis_gradient[ii] = error_norm_space_time(v_array_reference, \
                                                        v_array_dis_gradient[:, :, :, ii], dt_output, norm=norm_type)

        error_q_lin_implicit[ii] = error_norm_space_time(q_array_reference, \
                                                        q_array_lin_implicit[:, :, :, ii], dt_output, norm=norm_type)
        error_v_lin_implicit[ii] = error_norm_space_time(v_array_reference, \
                                                        v_array_lin_implicit[:, :, :, ii], dt_output, norm=norm_type)

        error_q_semiexplicit[ii] = error_norm_space_time(q_array_reference, \
                                                        q_array_semiexplicit[:, :, :, ii], dt_output, norm=norm_type)
        error_v_semiexplicit[ii] = error_norm_space_time(v_array_reference, \
                                                        v_array_semiexplicit[:, :, :, ii], dt_output, norm=norm_type)


dict_position = {"Discrete gradient": error_q_dis_gradient, 
                 "Linear implicit": error_q_lin_implicit,
                "Leapfrog": error_q_leapfrog}

dict_velocity = {"Discrete gradient": error_v_dis_gradient, 
                "Linear implicit": error_v_lin_implicit,
                "Leapfrog": error_v_leapfrog}

# dict_position = {"Discrete gradient": error_q_dis_gradient, 
#                  "Linear implicit": error_q_lin_implicit,
#                 "Leapfrog": error_q_leapfrog,
#                 "Semiexplicit": error_q_semiexplicit}

# dict_velocity = {"Discrete gradient": error_v_dis_gradient, 
#                 "Linear implicit": error_v_lin_implicit,
#                 "Leapfrog": error_v_leapfrog,
#                 "Semiexplicit": error_v_semiexplicit}

str_xlabel = '$\log \Delta t \; \mathrm{[s]}$'
plot_convergence(time_step_vec, dict_position, rate=True, xlabel=str_xlabel, ylabel=r"$\log \epsilon_{\bm q}$", \
                title=r'Error displacement $\bm{q}$', savefig=f"{directory_images}convergence_position")

plot_convergence(time_step_vec, dict_velocity, rate=True, xlabel=str_xlabel, ylabel=r"$\log \epsilon_{\bm v}$",  \
                 title=r'Error velocity $\bm{v}$', savefig=f"{directory_images}convergence_velocity")



plt.figure()
plt.loglog(time_step_vec, comp_time_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, comp_time_lin_implicit, '+-', label='Linear implicit')
plt.loglog(time_step_vec, comp_time_leapfrog, '^-', label='Leapfrog')
# plt.loglog(time_step_vec, comp_time_semiexplicit, 's-', label='Semiexplicit')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel(r"$\log T_{\rm comp}$")
plt.legend()
plt.grid(True)
plt.title("Computational time [s]")
plt.savefig(f"{directory_images}computational_time.pdf", dpi='figure', format='pdf', bbox_inches="tight")


plt.figure()
plt.loglog(time_step_vec, avg_diff_E_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, avg_diff_E_lin_implicit, '+-', label='Linear implicit')
plt.loglog(time_step_vec, avg_diff_E_leapfrog, '^-', label='Leapfrog')
# plt.loglog(time_step_vec, avg_diff_E_semiexplicit, 's-', label='Semiexplicit')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel(r"$\frac{1}{N_t}\sum_{n=0}^{N_t}|H_{n+1} - H_{n}|$")
plt.legend()
plt.grid(True)
plt.title("Mean of energy difference")
plt.savefig(f"{directory_images}energy_difference.pdf", dpi='figure', format='pdf', bbox_inches="tight")


plt.show()
