import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_vector_triplot
import matplotlib.pyplot as plt
from src.solvers.dynamics.hamiltonian_st_venant import HamiltonianElasticitySolver
from src.solvers.dynamics.nonlinear_lagrangian import NonlinearLagrangianSolver

from src.tools.common import compute_min_max_mesh

from src.problems.cantilever_beam import CantileverBeam
from src.problems.dynamic_cook_membrane import DynamicCookMembrane

import os

# # Stable choice non linear Lagrangian
pol_degree = 1

time_step = 1e-2

# pol_degree = 1
# quad = False
# n_elem_x= 100
# n_elem_y = 10
# time_step = 1e-2

T_end = 10
n_time  = ceil(T_end/time_step)

# quad = False
# n_elem_x= 100
# n_elem_y = 10
# problem = CantileverBeam(n_elem_x, n_elem_y, quad)

problem = DynamicCookMembrane(mesh_size=2)

solver = HamiltonianElasticitySolver(problem, 
                            time_step, 
                            pol_degree)

# solver = NonlinearLagrangianSolver(problem, 
#                                 time_step, 
#                                 pol_degree,
#                                 solver_parameters={})


if isinstance(solver, HamiltonianElasticitySolver):
    cfl_wave = solver.get_wave_cfl()
    print(f"CFL static value {cfl_wave}")

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
            os.makedirs(directory_results)
            
time_vector = np.linspace(0, T_end, num=n_time+1)
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(solver.energy_old)

power_balance_vector = np.zeros((n_time, ))

output_frequency = 10
displaced_mesh= solver.output_displaced_mesh()
displaced_coordinates_x = displaced_mesh.coordinates.dat.data[:, 0]
displaced_coordinates_y = displaced_mesh.coordinates.dat.data[:, 1]

min_max_coords_x = (min(displaced_coordinates_x), max(displaced_coordinates_x))
min_max_coords_y = (min(displaced_coordinates_y), max(displaced_coordinates_y))
list_min_max_coords = [min_max_coords_x, min_max_coords_y]
list_frames = []
time_frames = []
list_frames.append(displaced_mesh)
time_frames.append(0)


for ii in tqdm(range(1, n_time+1)):
    actual_time = ii*time_step

    solver.integrate()

    energy_vector[ii] = fdrk.assemble(solver.energy_new)

    if isinstance(solver, HamiltonianElasticitySolver):
        # print(f"Worst case CFL {cfl_wave + solver.get_dinamic_cfl()}")
        power_balance_vector[ii-1] = fdrk.assemble(solver.power_balance)

    solver.update_variables()

    if ii % output_frequency == 0:

        displaced_mesh = solver.output_displaced_mesh()
        list_min_max_coords = compute_min_max_mesh(displaced_mesh, list_min_max_coords)

        list_frames.append(displaced_mesh)
        time_frames.append(actual_time)


interval = 1e3 * output_frequency * time_step

lim_x, lim_y  = list_min_max_coords

animation = animate_vector_triplot(list_frames, interval, \
                                            lim_x = lim_x, \
                                            lim_y = lim_y )

animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")

n_frames = len(list_frames)-1

indexes_images = [int(n_frames/4), int(n_frames/2), int(3*n_frames/4), int(n_frames)]

for kk in indexes_images:
    time_image = "{:.1f}".format(time_step * output_frequency * kk) 

    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    fdrk.triplot(list_frames[kk], axes=axes)
    axes.set_title(f"Displacement at time $t={time_image}$" + r"$[\mathrm{s}]$", loc='center')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_xlim(lim_x)
    axes.set_ylim(lim_y)

    plt.savefig(f"{directory_results}Displacement_t{time_image}.eps", bbox_inches='tight', dpi='figure', format='eps')


plt.figure()
plt.plot(time_vector, energy_vector)
# plt.plot(time_vector, energy_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Energy")
plt.savefig(f"{directory_results}Energy.eps", dpi='figure', format='eps')

if isinstance(solver, HamiltonianElasticitySolver):
    plt.figure()
    plt.plot(time_vector[1:], np.diff(energy_vector) - time_step * power_balance_vector)
    # plt.plot(time_vector[1:], np.diff(energy_vector_linear) - power_balance_vector_linear, label=f"Linear")
    plt.grid(color='0.8', linestyle='-', linewidth=.5)
    plt.xlabel(r'Time')
    plt.legend()
    plt.title("Power balance conservation")
    plt.savefig(f"{directory_results}Power.eps", dpi='figure', format='eps')

plt.show()