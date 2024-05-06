import firedrake as fdrk
import numpy as np
from src.postprocessing.animators import animate_vector_triplot
import matplotlib.pyplot as plt
from src.solvers.dynamics.hamiltonian_neo_hooken import HamiltonianNeoHookeanSolver
from src.tools.common import compute_min_max_mesh
from src.problems.twisting_column import TwistingColumn
from src.problems.bending_column import BendingColumn
import os
from firedrake.petsc import PETSc

pol_degree = 1
T_end = 2

# problem = TwistingColumn(n_elem_x=6, n_elem_y=6, n_elem_z=36)
problem = BendingColumn(n_elem_x=12, n_elem_y=12, n_elem_z=72)

solver = HamiltonianNeoHookeanSolver(problem, 
                                    pol_degree)


directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
            os.makedirs(directory_results)

home_dir =os.environ['HOME']
directory_largedata = f"{home_dir}/StoreResults/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_largedata):
    os.makedirs(directory_largedata, exist_ok=True)

outfile_displacement = fdrk.File(f"{directory_largedata}/Displacement.pvd")
outfile_displacement.write(solver.displacement_old, time=0)
            
time_vector = []
time_vector.append(0)
energy_vector = []
energy_vector.append(fdrk.assemble(solver.energy(solver.velocity_old, solver.strain_old)))

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

ii = 0
actual_time = float(solver.actual_time_energy)
while actual_time < T_end:

    solver.integrate()

    energy_vector.append(fdrk.assemble(solver.energy(solver.velocity_new, solver.strain_new)))

    solver.update_variables()

    actual_time = float(solver.actual_time_energy)

    time_vector.append(actual_time)

    PETSc.Sys.Print(f"Actual time {actual_time:.3f}. Percentage : {actual_time/T_end*100:.1f}%")

    ii+=1
    if ii % output_frequency == 0:

        displaced_mesh = solver.output_displaced_mesh()
        list_min_max_coords = compute_min_max_mesh(displaced_mesh, list_min_max_coords)

        list_frames.append(displaced_mesh)
        time_frames.append(actual_time)

        outfile_displacement.write(solver.displacement_old, time=actual_time)

average_time_step = np.mean(np.diff(np.array(time_frames)))
interval = 10**2 * output_frequency * average_time_step
print(f"Interval {interval}")
lim_x, lim_y, lim_z  = list_min_max_coords

print(f"Lim x : {lim_x}")
print(f"Lim y : {lim_y}")
print(f"Lim z : {lim_z}")

animation = animate_vector_triplot(list_frames, interval, \
                                    lim_x = lim_x, \
                                    lim_y = lim_y, \
                                    lim_z = lim_z, three_dim=True)

animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")


n_frames = len(time_frames)
indexes_images = [0, int(n_frames/2), int(n_frames-1)]

for kk in indexes_images:
    time_image = time_frames[kk]

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.set_aspect('equal')
    fdrk.triplot(list_frames[kk], axes=axes)
    axes.set_title(f"Displacement $t={time_image:.1f}$ [ms]", loc='center')
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_xlim(lim_x)
    axes.set_ylim(lim_y)
    axes.set_zlim(lim_z)

    plt.savefig(f"{directory_results}/Displacement_t{time_image:.1f}.pdf", bbox_inches='tight', dpi='figure', format='pdf')


plt.figure()
plt.plot(time_vector, energy_vector)
# plt.plot(time_vector, energy_vector_linear, label=f"Linear")
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel('Time')
plt.legend()
plt.title("Energy")
plt.savefig(f"{directory_results}Energy.pdf", bbox_inches='tight', dpi='figure', format='pdf')

plt.show()