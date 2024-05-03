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

pol_degree = 2
time_step = 1e-4
T_end = 10*time_step
n_time  = ceil(T_end/time_step)

problem = TwistingColumn(n_elem_x=4, n_elem_y=4, n_elem_z=24)

solver = HamiltonianNeoHookeanSolver(problem, 
                            time_step, 
                            pol_degree)

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
            os.makedirs(directory_results)
            
time_vector = np.linspace(0, T_end, num=n_time+1)
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(solver.energy(solver.velocity_old, solver.strain_old))

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

    energy_vector[ii] = fdrk.assemble(solver.energy(solver.velocity_new, solver.strain_new))

    solver.update_variables()

    if ii % output_frequency == 0:

        displaced_mesh = solver.output_displaced_mesh()
        list_min_max_coords = compute_min_max_mesh(displaced_mesh, list_min_max_coords)

        list_frames.append(displaced_mesh)
        time_frames.append(actual_time)
