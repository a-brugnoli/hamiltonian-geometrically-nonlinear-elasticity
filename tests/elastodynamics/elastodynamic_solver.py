import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_displacement
import matplotlib.pyplot as plt
from src.solvers.implicit_linear_solver import ImplicitLinearSolver
from src.problems.cantilever_beam import CantileverBeam


pol_degree = 1
quad = False
n_elem_x= 100
n_elem_y = 10
time_step = 1e-2
T_end = 10
n_time  = ceil(T_end/time_step)


cantilever_beam = CantileverBeam(n_elem_x, n_elem_y, quad)

solver = ImplicitLinearSolver(cantilever_beam, 
                                "Elastodynamics", 
                                time_step, 
                                pol_degree)


time_vector = np.linspace(0, T_end, num=n_time+1)
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(solver.energy_old)

power_balance_vector = np.zeros((n_time, ))

output_frequency = 10

space_displacement = solver.operators.space_energy.sub(0)
displaced_coordinates = fdrk.interpolate(solver.problem.coordinates_mesh + solver.displacement_old, space_displacement)

displaced_mesh= fdrk.Mesh(displaced_coordinates)

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

        displaced_coordinates = fdrk.interpolate(solver.problem.coordinates_mesh \
                                                    + solver.displacement_old, space_displacement)
        
        displaced_mesh = fdrk.Mesh(displaced_coordinates)

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

animation.save(f"cantilever_quadmesh_{quad}.mp4", writer="ffmpeg")

n_frames = len(list_frames)-1

indexes_images = [int(n_frames/4), int(n_frames/2), int(3*n_frames/4), int(n_frames)]

for kk in indexes_images:

    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    fdrk.triplot(list_frames[kk], axes=axes)
    # axes.set_title(f"Displacement at time $t={time_image}$" + r"$[\mathrm{s}]$", loc='center')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_xlim(lim_x)
    axes.set_ylim(lim_y)

    plt.savefig(f"Displacement_index_{kk}.eps", bbox_inches='tight', dpi='figure', format='eps')


