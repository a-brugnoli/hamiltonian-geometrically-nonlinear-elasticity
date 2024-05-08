import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_scalar_trisurf, animate_scalar_tripcolor
from src.tools.common import compute_min_max_function
import matplotlib.pyplot as plt
from src.solvers.dynamics.hamiltonian_von_karman import HamiltonianVonKarmanSolver
from src.problems.dynamics.free_vibrations_von_karman import FirstModeVonKarman
import os, sys

n_elem = 30
pol_degree = 1
quad = False
time_step = 5*10**(-6)
T_end = 7.5 * 10**(-3)
# T_end = 30 * time_step

n_time  = ceil(T_end/time_step)

amplitude = 10
problem = FirstModeVonKarman(n_elem, n_elem, amplitude=amplitude)

solver = HamiltonianVonKarmanSolver(problem, 
                            time_step, 
                            pol_degree, \
                            coupling=True, 
                            membrane_inertia=False)

absolute_path = os.path.dirname(os.path.abspath(__file__))
append_to_result_folder = f"n_elem_{n_elem}_deg_{pol_degree}/amp_{amplitude}/"

directory_results = f"{absolute_path}/results/{str(solver)}/{str(problem)}/{append_to_result_folder}"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)
            
time_vector_ms = np.linspace(0, T_end, num=n_time+1)*1000
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(solver.energy_old)

L_x, L_y = problem.parameters["L_x"], problem.parameters["L_y"]
center = (L_x/2, L_y/2)

displacement_at_center = np.zeros((n_time+1, ))
displacement_at_center[0] = solver.bend_displacement_old.at(center)

velocity_at_center = np.zeros((n_time+1, ))
velocity_at_center[0] = solver.bend_velocity_old.at(center)

output_frequency = 10

min_max_vel = (0, 0)
min_max_disp = (0, 0)

time_frames_ms = []
time_frames_ms.append(0)

list_frames_bend_velocity = []
list_frames_bend_velocity.append(solver.bend_velocity_old.copy(deepcopy=True))

list_frames_bend_displacement = []
list_frames_bend_displacement.append(solver.bend_displacement_old.copy(deepcopy=True))

home_dir =os.environ['HOME']
directory_largedata = f"{home_dir}/StoreResults/{str(solver)}/{str(problem)}/{append_to_result_folder}"
if not os.path.exists(directory_largedata):
    os.makedirs(directory_largedata, exist_ok=True)

outfile_bend_velocity = fdrk.File(f"{directory_largedata}/Vertical_velocity.pvd")
outfile_bend_velocity.write(solver.bend_velocity_old, time=0)

outfile_bend_displacement = fdrk.File(f"{directory_largedata}/Vertical_displacement.pvd")
outfile_bend_displacement.write(solver.bend_displacement_old, time=0)

for ii in tqdm(range(1, n_time+1)):
    actual_time = ii*time_step

    solver.integrate()
    energy_vector[ii] = fdrk.assemble(solver.energy_new)

    # residual_displacement = solver.bend_displacement_new - solver.bend_displacement_old \
    #                         - time_step * solver.bend_velocity_new

    # print(f"Residual displacement eq {fdrk.norm(residual_displacement)}")

    solver.update_variables()
    displacement_at_center[ii] = solver.bend_displacement_old.at(center)
    velocity_at_center[ii] = solver.bend_velocity_old.at(center)

    if ii % output_frequency == 0:

        min_max_vel = compute_min_max_function(solver.bend_velocity_old, min_max_vel)
        min_max_disp = compute_min_max_function(solver.bend_displacement_old, min_max_disp)

        list_frames_bend_velocity.append(solver.bend_velocity_old.copy(deepcopy=True))
        list_frames_bend_displacement.append(solver.bend_displacement_old.copy(deepcopy=True))
        time_frames_ms.append(10**3 * actual_time)

        outfile_bend_velocity.write(solver.bend_velocity_old, time=actual_time)
        outfile_bend_displacement.write(solver.bend_displacement_old, time=actual_time)



interval = 10**6 * output_frequency * time_step

velocity_animation = animate_scalar_trisurf(time_frames_ms, list_frames_bend_velocity,\
                        interval=interval, lim_z = min_max_vel, \
                        title = "Velocity", xlabel= r"$x [\mathrm{m}]$", ylabel = r"$y [\mathrm{m}]$")


velocity_animation.save(f"{directory_results}Animation_velocity.mp4", writer="ffmpeg")

displacement_animation = animate_scalar_trisurf(time_frames_ms, list_frames_bend_displacement,\
                        interval=interval, lim_z = min_max_disp, \
                        title = "Displacement", xlabel= r"$x [\mathrm{m}]$", ylabel = r"$y [\mathrm{m}]$")

displacement_animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")

n_frames = len(time_frames_ms)
indexes_images = [0, int(n_frames/3), int(2*n_frames/3), int(n_frames-1)]

for kk in indexes_images:
    time_image = time_frames_ms[kk]

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.set_aspect('equal')
    fdrk.trisurf(list_frames_bend_displacement[kk], axes=axes)
    axes.set_title(f"$w(t={time_image:.1f}$" + r"$\; [\mathrm{ms}]$)", loc='center')
    axes.set_xlabel(r"$x [\mathrm{m}]$")
    axes.set_ylabel(r"$y [\mathrm{m}]$")
    axes.set_zlim(min_max_disp)

    plt.savefig(f"{directory_results}/Displacement_t{time_image:.1f}.pdf", \
                bbox_inches='tight', pad_inches=0.3, dpi='figure', format='pdf')

plt.figure()
plt.plot(time_vector_ms, energy_vector)
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel(r"Time $[\mathrm{ms}]$")
plt.title("Energy")
plt.savefig(f"{directory_results}/Energy.pdf", dpi='figure', format='pdf')

plt.close('all')
