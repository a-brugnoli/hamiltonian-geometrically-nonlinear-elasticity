from cantilever_beam import simulate_cantilever_beam
# from cantilever_beam_nedelec import simulate_cantilever_beam
import src.postprocessing.options
import matplotlib.pyplot as plt
import numpy as np

isquad = False
time_vector, energy_vector_nonlinear, power_balance_vector_nonlinear = \
    simulate_cantilever_beam(is_quad_mesh=isquad, linear=False, pol_degree=1, n_elem_x=100)

# time_vector, energy_vector_linear, power_balance_vector_linear = \
#     simulate_cantilever_beam(is_quad_mesh=isquad, linear=True)


plt.figure()
plt.plot(time_vector, energy_vector_nonlinear, label=f"Non Linear")
# plt.plot(time_vector, energy_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Energy")
plt.savefig(f"Energy_cantilever_quad_{isquad}.eps", dpi='figure', format='eps')


plt.figure()
plt.plot(time_vector[1:], np.diff(energy_vector_nonlinear) - power_balance_vector_nonlinear, label=f"Non linear")
# plt.plot(time_vector[1:], np.diff(energy_vector_linear) - power_balance_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Power balance conservation")
plt.savefig(f"Power_cantilever_quad_{isquad}.eps", dpi='figure', format='eps')

# plt.show()