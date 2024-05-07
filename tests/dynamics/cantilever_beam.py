import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_vector_triplot
import matplotlib.pyplot as plt
from src.solvers.dynamics.hamiltonian_st_venant import HamiltonianSaintVenantSolver
from src.solvers.dynamics.nonlinear_lagrangian import NonlinearLagrangianSolver

from src.tools.common import compute_min_max_mesh

from src.problems.dynamics.cantilever_beam import CantileverBeam
from src.problems.dynamics.dynamic_cook_membrane import DynamicCookMembrane
import time
from firedrake.petsc import PETSc


import os

# # Stable choice non linear Lagrangian
pol_degree = 1

# pol_degree = 1
# quad = False
# n_elem_x= 100
# n_elem_y = 10

T_end = 10
# time_step = 1e-2
# n_time  = ceil(T_end/time_step)

quad = False
n_elem_x= 100
n_elem_y = 10
problem = CantileverBeam(n_elem_x, n_elem_y, quad)

# problem = DynamicCookMembrane(mesh_size=2)

solver = HamiltonianSaintVenantSolver(problem, pol_degree)

# solver = NonlinearLagrangianSolver(problem, 
#                                 time_step, 
#                                 pol_degree,
#                                 solver_parameters={})


directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
            os.makedirs(directory_results)
            

time_vector = []
time_vector.append(0)
energy_vector = []
energy_vector.append(fdrk.assemble(solver.energy_old))
time_step_vec = []
power_balance_vector = []

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

ii = 0
actual_time = float(solver.actual_time_energy)
computing_time = 0

while actual_time < T_end:

    start_iteration = time.time()
    solver.integrate()
    end_iteration = time.time()
    elapsed_iteration = end_iteration - start_iteration
    computing_time += elapsed_iteration

    energy_vector.append(fdrk.assemble(solver.energy_new))
    time_step_vec.append(float(solver.time_step))

    if isinstance(solver, HamiltonianSaintVenantSolver):
        # print(f"Worst case CFL {cfl_wave + solver.get_dinamic_cfl()}")
        power_balance_vector.append(fdrk.assemble(solver.power_balance))

    solver.update_variables()

    actual_time = float(solver.actual_time_energy)
    time_vector.append(actual_time)
    time_fraction = actual_time/T_end

    expected_total_computing_time = computing_time/time_fraction
    expected_remaining_time = expected_total_computing_time - computing_time
    ii+=1
    PETSc.Sys.Print(f"Iteration number {ii}. Actual time {actual_time:.3f}. Percentage : {time_fraction*100:.1f}%")
    PETSc.Sys.Print(f"Expected time to end : {expected_remaining_time:.1f} (s).")


    if ii % output_frequency == 0:

        displaced_mesh = solver.output_displaced_mesh()
        list_min_max_coords = compute_min_max_mesh(displaced_mesh, list_min_max_coords)

        list_frames.append(displaced_mesh)
        time_frames.append(actual_time)


interval = 1e3 * output_frequency * sum(time_step_vec)/len(time_step_vec)

lim_x, lim_y  = list_min_max_coords

animation = animate_vector_triplot(list_frames, interval, \
                                            lim_x = lim_x, \
                                            lim_y = lim_y )

animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")

n_frames = len(list_frames)-1

indexes_images = [int(n_frames/4), int(n_frames/2), int(3*n_frames/4), int(n_frames)]

for kk in indexes_images:
    time_image = "{:.1f}".format(time_vector[kk] * output_frequency) 

    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    fdrk.triplot(list_frames[kk], axes=axes)
    axes.set_title(f"Displacement at time $t={time_image}$ [s].", loc='center')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_xlim(lim_x)
    axes.set_ylim(lim_y)

    plt.savefig(f"{directory_results}Displacement_t{time_image}.eps", bbox_inches='tight', dpi='figure', format='eps')


plt.figure()
plt.plot(time_vector, energy_vector)
# plt.plot(time_vector, energy_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel('Time')
plt.legend()
plt.title("Energy")
plt.savefig(f"{directory_results}Energy.eps", dpi='figure', format='eps')



if isinstance(solver, HamiltonianSaintVenantSolver):
    plt.figure()
    dt_power = [dt * power_t for dt, power_t in zip(time_step_vec, power_balance_vector)]
    plt.plot(time_vector[1:], np.diff(energy_vector) - dt_power)
    # plt.plot(time_vector[1:], np.diff(energy_vector_linear) - power_balance_vector_linear, label=f"Linear")
    plt.grid(color='0.8', linestyle='-', linewidth=.5)
    plt.xlabel('Time')
    plt.legend()
    plt.title("Power balance conservation")
    plt.savefig(f"{directory_results}Power.eps", dpi='figure', format='eps')

plt.show()