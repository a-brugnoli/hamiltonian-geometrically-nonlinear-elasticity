import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_displacement
import matplotlib.pyplot as plt
# from src.solvers.hamiltonian_solver import HamiltonianSolver
from src.solvers.hamiltonian_displacement_solver import HamiltonianDisplacementSolver
from src.problems.cantilever_beam import CantileverBeam
import os

pol_degree = 1
quad = False
n_elem_x= 100
n_elem_y = 10
time_step = 1e-2
T_end = 10
n_time  = ceil(T_end/time_step)

problem = CantileverBeam(n_elem_x, n_elem_y, quad)

# solver = HamiltonianSolver(problem, 
#                             "Elastodynamics", 
#                             time_step, 
#                             pol_degree)

solver = HamiltonianDisplacementSolver(problem, 
                            time_step, 
                            pol_degree)


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

min_x_all = min(displaced_coordinates_x)
max_x_all = max(displaced_coordinates_x)

displaced_coordinates_y = displaced_mesh.coordinates.dat.data[:, 1]

min_y_all = min(displaced_coordinates_y)
max_y_all = max(displaced_coordinates_y)


list_frames = []
time_frames = []
list_frames.append(displaced_mesh)
time_frames.append(0)


for ii in tqdm(range(1, n_time+1)):
    actual_time = ii*time_step

    solver.integrate()

    energy_vector[ii] = fdrk.assemble(solver.energy_new)
    power_balance_vector[ii-1] = fdrk.assemble(solver.power_balance)

    solver.update_variables()


    if ii % output_frequency == 0:

        displaced_mesh = solver.output_displaced_mesh()

        displaced_coordinates_x = displaced_mesh.coordinates.dat.data[:, 0]

        min_x = min(displaced_coordinates_x)
        max_x = max(displaced_coordinates_x)

        if min_x<min_x_all:
            min_x_all = min_x
        if max_x>max_x_all:
            max_x_all = max_x

        displaced_coordinates_y = displaced_mesh.coordinates.dat.data[:, 1]

        min_y = min(displaced_coordinates_y)
        max_y = max(displaced_coordinates_y)
        
        if min_y<min_y_all:
            min_y_all = min_y
        if max_y>max_y_all:
            max_y_all = max_y


        list_frames.append(displaced_mesh)
        time_frames.append(actual_time)


interval = 1e3 * output_frequency * time_step

lim_x = (min_x_all, max_x_all)
lim_y = (min_y_all, max_y_all)

animation = animate_displacement(time_frames, list_frames, interval, \
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


plt.figure()
plt.plot(time_vector[1:], np.diff(energy_vector) - time_step * power_balance_vector)
# plt.plot(time_vector[1:], np.diff(energy_vector_linear) - power_balance_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Power balance conservation")
plt.savefig(f"{directory_results}Power.eps", dpi='figure', format='eps')

plt.show()