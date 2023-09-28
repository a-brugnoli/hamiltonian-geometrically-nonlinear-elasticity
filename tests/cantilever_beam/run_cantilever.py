from cantilever_beam import simulate_cantilever_beam
# from cantilever_beam_nedelec import simulate_cantilever_beam
import src.postprocessing.options
import matplotlib.pyplot as plt
import numpy as np
import firedrake as fdrk

time_vector, energy_vector_nonlinear, power_balance_vector_nonlinear, \
    output_freq, indexes_images, list_images_nonlinear = simulate_cantilever_beam(is_quad_mesh=False, \
                                                                                                linear=False)
time_vector, energy_vector_linear, power_balance_vector_linear, \
    _, _, list_images_linear = simulate_cantilever_beam(is_quad_mesh=False, linear=True)

for kk in range(len(indexes_images)):
    time_image = time_vector[output_freq*indexes_images[kk]]
    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    triplot_lin = fdrk.triplot(list_images_linear[kk], axes=axes)
    # axes.set_title(f"Displacement at time $t={time_image}$" + r"$[\mathrm{s}]$", loc='center')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    plt.savefig(f"Displacement_linear_{indexes_images[kk]}.eps", bbox_inches='tight', dpi='figure', format='eps')

    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    triplot_nonlin =fdrk.triplot(list_images_nonlinear[kk], axes=axes)
    # axes.set_title(f"Displacement at time $t={time_image}$" + r"$[\mathrm{s}]$", loc='center')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    plt.savefig(f"Displacement_nonlinear_{indexes_images[kk]}.eps", bbox_inches='tight', dpi='figure', format='eps')



plt.figure()
plt.plot(time_vector, energy_vector_nonlinear, label=f"Non Linear")
plt.plot(time_vector, energy_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Energy")
plt.savefig("Energy_cantilever.eps", dpi='figure', format='eps')


plt.figure()
plt.plot(time_vector[1:], np.diff(energy_vector_nonlinear) - power_balance_vector_nonlinear, label=f"Non linear")
plt.plot(time_vector[1:], np.diff(energy_vector_linear) - power_balance_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Power balance conservation")
plt.savefig("Power_cantilever.eps", dpi='figure', format='eps')

plt.show()