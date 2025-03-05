import matplotlib.pyplot as plt
from src.postprocessing.plot_convergence import plot_convergence
from experiments.duffing.parameters_duffing import *
import pickle
from src.postprocessing.options import configure_matplotib
configure_matplotib()


with open(file_results_error_position, "rb") as f:
        dict_error_position = pickle.load(f)

str_xlabel = '$\log \Delta t \; \mathrm{[s]}$'
plot_convergence(time_step_vec, dict_error_position, rate=True, xlabel=str_xlabel, ylabel="$\log \epsilon_q$", \
                title='Error position $q$', savefig=f"{directory_images}convergence_position")

with open(file_results_error_velocity, "rb") as f:
        dict_error_velocity = pickle.load(f)

plot_convergence(time_step_vec, dict_error_velocity,  rate=True, xlabel=str_xlabel, ylabel="$\log \epsilon_v$",  \
                 title='Error velocity $v$', savefig=f"{directory_images}convergence_velocity")

with open(file_results_error_energy, "rb") as f:
        dict_error_energy = pickle.load(f)

error_vec_E_leapfrog = dict_error_energy["Leapfrog"]
error_vec_E_dis_gradient = dict_error_energy["Discrete gradient"]
error_vec_E_lin_implicit = dict_error_energy["Linear implicit"]

plt.figure()
plt.loglog(time_step_vec, error_vec_E_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, error_vec_E_lin_implicit, '+--', label='Linear implicit')
plt.loglog(time_step_vec, error_vec_E_leapfrog, '^:', label='Leapfrog')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel("$\log \epsilon_H$")

plt.legend()
plt.grid(True)
plt.title("Error energy")
plt.savefig(f"{directory_images}energy_error.pdf", dpi='figure', format='pdf', bbox_inches="tight")

with open(file_results_comp_time, "rb") as f:
        dict_comp_time = pickle.load(f)

elapsed_vec_leapfrog = dict_comp_time["Leapfrog"]
elapsed_vec_dis_gradient = dict_comp_time["Discrete gradient"]
elapsed_vec_lin_implicit = dict_comp_time["Linear implicit"]

plt.figure()
plt.loglog(time_step_vec, elapsed_vec_dis_gradient, 'o-', label='Discrete gradient')
plt.loglog(time_step_vec, elapsed_vec_lin_implicit, '+--', label='Linear implicit')
plt.loglog(time_step_vec, elapsed_vec_leapfrog, '^:', label='Leapfrog')
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(str_xlabel)
plt.ylabel(r"$\log T_{\rm comp}$")
plt.legend()
plt.grid(True)
plt.title("Computational time [ms]")
plt.savefig(f"{directory_images}computational_time.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.show()