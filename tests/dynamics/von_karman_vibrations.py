import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_scalar_trisurf, animate_scalar_tripcolor
import matplotlib.pyplot as plt
from src.solvers.hamiltonian_von_karman import HamiltonianVonKarmanSolver
from src.problems.free_vibrations_von_karman import FirstModeVonKarman
import os

pol_degree = 1
quad = False
n_elem_x = 10
n_elem_y = 10
time_step = 5*10**(-6)
T_end = 10 * time_step
# T_end = 7.5*10**(-3)

n_time  = ceil(T_end/time_step)

problem = FirstModeVonKarman(n_elem_x, n_elem_y)

solver = HamiltonianVonKarmanSolver(problem, 
                            time_step, 
                            pol_degree)


directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
            os.makedirs(directory_results)
            
time_vector = np.linspace(0, T_end, num=n_time+1)
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(solver.energy_old)

output_frequency = 1

min_z_all = 0
max_z_all = 0

list_frames_velocity = []
list_frames_velocity.append(solver.bend_velocity_old)
time_frames_ms = []
time_frames_ms.append(0)

directory_largedata = "/home/dmsm/a.brugnoli/StoreResults/VonKarman/"
if not os.path.exists(directory_largedata):
    os.makedirs(directory_largedata, exist_ok=True)

outfile_bend_velocity = fdrk.File(f"{directory_largedata}{str(solver)}/Vertical_velocity.pvd")
outfile_bend_velocity.write(solver.bend_velocity_old, time=0)


for ii in tqdm(range(1, n_time+1)):
    actual_time = ii*time_step

    solver.integrate()

    energy_vector[ii] = fdrk.assemble(solver.energy_new)

    solver.update_variables()

    if ii % output_frequency == 0:

        bend_velocity_vector = solver.bend_velocity_old.vector().get_local()

        min_z = min(bend_velocity_vector)
        max_z = max(bend_velocity_vector)

        if min_z<min_z_all:
            min_z_all = min_z
        if max_z>max_z_all:
            max_z_all = max_z

        list_frames_velocity.append(solver.bend_velocity_old.copy(deepcopy=True))
        time_frames_ms.append(10e6 * actual_time)

        outfile_bend_velocity.write(solver.bend_velocity_old, time=actual_time)

interval = 10e6 * output_frequency * time_step
# velocity_animation = animate_scalar_tripcolor(problem.domain, list_frames_velocity, interval=interval)

velocity_animation = animate_scalar_trisurf(time_frames_ms, list_frames_velocity,\
                                            interval=interval, lim_z=(min_z_all, max_z_all))

velocity_animation.save(f"{directory_results}Animation_velocity.mp4", writer="ffmpeg")

# plt.figure()
# plt.plot(time_vector, energy_vector)
# plt.grid(color='0.8', linestyle='-', linewidth=.5)
# plt.xlabel(r'Time')
# plt.legend()
# plt.title("Energy")
# plt.savefig(f"{directory_results}Energy.eps", dpi='figure', format='eps')

# plt.show()