from cantilever_beam import simulate_cantilever_beam
import src.postprocessing.options
import matplotlib.pyplot as plt
import numpy as np

time_vector, energy_vector_nonlinear, power_balance_vec_nonlinear = simulate_cantilever_beam(False)
time_vector, energy_vector_linear, power_balance_vec_linear = simulate_cantilever_beam(True)

plt.figure()
plt.plot(time_vector, energy_vector_nonlinear, label=f"Non Linear")
plt.plot(time_vector, energy_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Energy")
plt.savefig("Energy_cantilever.eps", dpi='figure', format='eps')


plt.figure()
plt.plot(time_vector[1:], np.diff(energy_vector_nonlinear) - power_balance_vec_nonlinear, label=f"Linear")
plt.plot(time_vector[1:], np.diff(energy_vector_linear) - power_balance_vec_linear, label=f"Non linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Power balance conservation")
plt.savefig("Power_cantilever.eps", dpi='figure', format='eps')
plt.show()