import time
import pickle
import numpy as np
from experiments.finite_strain_elasticity.parameters_elasticity import *
import matplotlib.pyplot as plt
from src.postprocessing.options import configure_matplotib
configure_matplotib()

bending_column.set_time_step(dt_base/4)
dict_results_leapfrog = bending_column.leapfrog(save_vars=True)
angular_momentum_leapfrog = dict_results_leapfrog["angular momentum"]

# dict_results_linear_implicit = bending_column.linear_implicit_static_condensation(save_vars=True,\
#                                                         paraview_directory=paraview_directory)
# dict_results_linear_implicit = bending_column.linear_implicit_static_condensation(save_vars=True)
# angular_momentum_linear_implicit = dict_results_linear_implicit["angular momentum"]

dict_results_dis_gradient = bending_column.implicit_method(method="discrete gradient", save_vars=True)
angular_momentum_dis_gradient = dict_results_dis_gradient["angular momentum"]

plt.figure()
plt.plot(t_vec_output, angular_momentum_leapfrog, '+', label='Leapfrog')
# plt.plot(t_vec_output, angular_momentum_linear_implicit, '--', label='Linear implicit')
plt.plot(t_vec_output, angular_momentum_dis_gradient, '-.', label='Discrete gradient')
plt.xlabel('Time')
plt.ylabel('J')
plt.legend()
plt.grid(True)
plt.title('Angular momentum vs Time')
plt.savefig(f"{directory_images}angular_momentum.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.show()