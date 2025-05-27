# import time
# import pickle
import numpy as np
from experiments.finite_strain_elasticity.parameters_elasticity import *
import matplotlib.pyplot as plt
from src.postprocessing.options import configure_matplotib
configure_matplotib()

test_ang_momentum = True

bending_column.set_time_step(dt_base/4)
dict_results_leapfrog = bending_column.leapfrog(save_vars=True, 
                                                test_angular_momentum=test_ang_momentum)
angular_momentum_leapfrog = dict_results_leapfrog["angular momentum"]

dict_results_linear_implicit = bending_column.linear_implicit_static_condensation(save_vars=True,
                                                        test_angular_momentum=test_ang_momentum)
angular_momentum_linear_implicit = dict_results_linear_implicit["angular momentum"]

dict_results_dis_gradient = bending_column.implicit_method(method="discrete gradient", save_vars=True,
                                                        test_angular_momentum=test_ang_momentum)
angular_momentum_dis_gradient = dict_results_dis_gradient["angular momentum"]

plt.figure()
plt.plot(t_vec_output[1:], np.diff(angular_momentum_leapfrog[:, 0]), '-+', label='Leapfrog')
plt.plot(t_vec_output[1:], np.diff(angular_momentum_linear_implicit[:, 0]), '--k', label='Linear implicit')
plt.plot(t_vec_output[1:], np.diff(angular_momentum_dis_gradient[:, 0]), '-.o', label='Discrete gradient')
plt.xlabel('Time [s]')
plt.ylabel('$J_x^{n+1} - J_x^n$')
plt.legend()
plt.grid(True)
plt.title('Difference Angular momentum $J_x$')
plt.savefig(f"{directory_images}angular_momentum_x.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.figure()
plt.plot(t_vec_output[1:], np.diff(angular_momentum_leapfrog[:, 1]), '-+', label='Leapfrog')
plt.plot(t_vec_output[1:], np.diff(angular_momentum_linear_implicit[:, 1]), '--k', label='Linear implicit')
plt.plot(t_vec_output[1:], np.diff(angular_momentum_dis_gradient[:, 1]), '-.o', label='Discrete gradient')
plt.xlabel('Time [s]')
plt.ylabel('$J_y^{n+1} - J_y^n$')
plt.legend()
plt.grid(True)
plt.title('Difference Angular momentum $J_y$')
plt.savefig(f"{directory_images}angular_momentum_y.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.figure()
plt.plot(t_vec_output[1:], np.diff(angular_momentum_leapfrog[:, 2]), '-+', label='Leapfrog')
plt.plot(t_vec_output[1:], np.diff(angular_momentum_linear_implicit[:, 2]), '--k', label='Linear implicit')
plt.plot(t_vec_output[1:], np.diff(angular_momentum_dis_gradient[:, 2]), '-.o', label='Discrete gradient')
plt.xlabel('Time [s]')
plt.ylabel('$J_z^{n+1} - J_z^n$')
plt.legend()
plt.grid(True)
plt.title('Difference Angular momentum $J_z$')
plt.savefig(f"{directory_images}angular_momentum_z.pdf", dpi='figure', format='pdf', bbox_inches="tight")

plt.show()