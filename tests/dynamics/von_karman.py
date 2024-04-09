import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_scalar_displacement
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

fdrk.trisurf(solver.bend_displacement_old)

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
            os.makedirs(directory_results)
            
time_vector = np.linspace(0, T_end, num=n_time+1)
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(solver.energy_old)

output_frequency = 10

min_z_all = 0
max_z_all = 0

list_frames = []
time_frames = []
list_frames.append(solver.bend_displacement_old)
time_frames.append(float(solver.actual_time_displacement))

for ii in tqdm(range(1, n_time+1)):
    actual_time = ii*time_step

    solver.integrate()

    energy_vector[ii] = fdrk.assemble(solver.energy_new)

    solver.update_variables()

    if ii % output_frequency == 0:

        bend_displacement_vector = solver.bend_displacement_old.vector().get_local()

        min_z = min(bend_displacement_vector)
        max_z = max(bend_displacement_vector)

        if min_z<min_z_all:
            min_z_all = min_z
        if max_z>max_z_all:
            max_z_all = max_z

        list_frames.append(solver.bend_displacement_old)
        time_frames.append(float(solver.actual_time_displacement))

print(f"Length list {len(list_frames)}")

# interval = 10e6 * output_frequency * time_step
# animation = animate_scalar_displacement(problem.domain, list_frames, interval=interval)
# animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")


directory_largedata = "/home/dmsm/a.brugnoli/StoreResults/VonKarman/"
if not os.path.exists(directory_largedata):
    os.makedirs(directory_largedata, exist_ok=True)


fdrk.trisurf(solver.bend_displacement_old)
plt.show()

outfile_bend_displacement = fdrk.File(f"{directory_largedata}{str(solver)}/Vertical_displacement.pvd")

for ii in range(len(list_frames)):
       outfile_bend_displacement.write(list_frames[ii], time=time_frames[ii])


# # n_frames = len(list_frames)-1
# # indexes_images = [int(n_frames/4), int(n_frames/2), int(3*n_frames/4), int(n_frames)]
# # for kk in indexes_images:
# #     time_image = "{:.6f}".format(time_step * output_frequency * kk) 

# #     fig = plt.figure()
# #     axes = fig.add_subplot(111, projection='3d')
# #     axes.set_aspect("equal")
# #     fdrk.trisurf(list_frames[kk], axes=axes)
# #     axes.set_title(f"Displacement at time $t={time_image}$" + r"$[\mathrm{s}]$", loc='center')
# #     axes.set_xlabel("x")
# #     axes.set_ylabel("y")
# #     axes.set_zlim(lim_z)

# #     plt.savefig(f"{directory_results}Displacement_t{time_image}.eps", bbox_inches='tight', dpi='figure', format='eps')

# # plt.figure()
# # plt.plot(time_vector, energy_vector)
# # plt.grid(color='0.8', linestyle='-', linewidth=.5)
# # plt.xlabel(r'Time')
# # plt.legend()
# # plt.title("Energy")
# # plt.savefig(f"{directory_results}Energy.eps", dpi='figure', format='eps')

# # plt.show()