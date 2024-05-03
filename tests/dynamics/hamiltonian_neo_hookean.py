import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_vector_triplot
import matplotlib.pyplot as plt
from src.solvers.dynamics.hamiltonian_neo_hooken import HamiltonianNeoHookeanSolver
from src.tools.common import compute_min_max_mesh
from src.problems.twisting_column import TwistingColumn

import os

pol_degree = 1
time_step = 1e-3
T_end = 10*time_step
n_time  = ceil(T_end/time_step)

problem = TwistingColumn(n_elem_x=12, n_elem_y=12, n_elem_z=72)

solver = HamiltonianNeoHookeanSolver(problem, 
                                    time_step, 
                                    pol_degree)

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
            os.makedirs(directory_results)

home_dir =os.environ['HOME']
directory_largedata = f"{home_dir}/StoreResults/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_largedata):
    os.makedirs(directory_largedata, exist_ok=True)

# outfile_displacement = fdrk.File(f"{directory_largedata}/Displacement.pvd")
# outfile_displacement.write(solver.displacement_old, time=0)
            
time_vector = np.linspace(0, T_end, num=n_time+1)
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(solver.energy(solver.velocity_old, solver.strain_old))

power_balance_vector = np.zeros((n_time, ))

output_frequency = 1
displaced_mesh= solver.output_displaced_mesh()
displaced_coordinates_x = displaced_mesh.coordinates.dat.data[:, 0]
displaced_coordinates_y = displaced_mesh.coordinates.dat.data[:, 1]
displaced_coordinates_z = displaced_mesh.coordinates.dat.data[:, 2]

min_max_coords_x = (min(displaced_coordinates_x), max(displaced_coordinates_x))
min_max_coords_y = (min(displaced_coordinates_y), max(displaced_coordinates_y))
min_max_coords_z = (min(displaced_coordinates_z), max(displaced_coordinates_z))

list_min_max_coords = [min_max_coords_x, min_max_coords_y, min_max_coords_z]
list_frames = []
time_frames = []
list_frames.append(displaced_mesh)
time_frames.append(0)


for ii in tqdm(range(1, n_time+1)):
    actual_time = ii*time_step

    solver.integrate()

    energy_vector[ii] = fdrk.assemble(solver.energy(solver.velocity_new, solver.strain_new))

    solver.update_variables()

    if ii % output_frequency == 0:

        displaced_mesh = solver.output_displaced_mesh()
        list_min_max_coords = compute_min_max_mesh(displaced_mesh, list_min_max_coords)

        list_frames.append(displaced_mesh)
        time_frames.append(actual_time)

        # outfile_displacement.write(solver.displacement_old, time=actual_time)


interval = 1e3 * output_frequency * time_step

lim_x, lim_y, lim_z  = list_min_max_coords

animation = animate_vector_triplot(list_frames, interval, \
                                    lim_x = lim_x, \
                                    lim_y = lim_y, \
                                    lim_z = lim_z, three_dim=True)

animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")


plt.figure()
plt.plot(time_vector, energy_vector)
# plt.plot(time_vector, energy_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r'Time')
plt.legend()
plt.title("Energy")
plt.savefig(f"{directory_results}Energy.eps", dpi='figure', format='eps')

plt.show()