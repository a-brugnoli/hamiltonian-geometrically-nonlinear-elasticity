import firedrake as fdrk
import matplotlib.pyplot as plt
from src.postprocessing.animators import animate_vector_triplot

from src.solvers.dynamics.hamiltonian_neo_hookean import HamiltonianNeoHookeanSolver
from src.solvers.dynamics.hamiltonian_st_venant import HamiltonianSaintVenantSolver
from src.tools.common import compute_min_max_mesh
from src.problems.dynamics.twisting_column import TwistingColumn
from src.problems.dynamics.bending_column import BendingColumn
import os
from firedrake.petsc import PETSc
import time

pol_degree = 1
T_end = 2

# problem = TwistingColumn(n_elem_x=6, n_elem_y=6, n_elem_z=36)
# problem = BendingColumn(n_elem_x=12, n_elem_y=12, n_elem_z=72)
problem = BendingColumn(n_elem_x=6, n_elem_y=6, n_elem_z=36)

# solver = HamiltonianNeoHookeanSolver(problem, 
#                                     pol_degree)

solver = HamiltonianSaintVenantSolver(problem, 
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
if isinstance(solver, HamiltonianNeoHookeanSolver):
    energy_vector.append(fdrk.assemble(solver.energy(solver.velocity_new, solver.strain_new)))
else:
    energy_vector.append(fdrk.assemble(solver.energy_old))

time_step_vec = []

output_frequency = 10

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
computing_time = 0

while actual_time < T_end:

    start_iteration = time.time()
    solver.integrate()
    end_iteration = time.time()
    elapsed_iteration = end_iteration - start_iteration
    computing_time += elapsed_iteration

    if isinstance(solver, HamiltonianNeoHookeanSolver):
        energy_vector.append(fdrk.assemble(solver.energy(solver.velocity_new, solver.strain_new)))
    else:
        energy_vector.append(fdrk.assemble(solver.energy_new))

    time_step_vec.append(float(solver.time_step))

    solver.update_variables()

    actual_time = float(solver.actual_time_energy)
    time_vector.append(actual_time)
    time_fraction = actual_time/T_end

    expected_total_computing_time = computing_time/time_fraction
    expected_remaining_time = expected_total_computing_time - computing_time
    ii+=1
    PETSc.Sys.Print(f"Iteration number {ii}. Actual time {actual_time:.3f}. Percentage : {time_fraction*100:.1f}%")
    PETSc.Sys.Print(f"Total computing time {computing_time:.1f}. Expected time to end : {expected_remaining_time/60:.1f} (min)")

    if ii % output_frequency == 0:

        displaced_mesh = solver.output_displaced_mesh()
        list_min_max_coords = compute_min_max_mesh(displaced_mesh, list_min_max_coords)

        list_frames.append(displaced_mesh)
        time_frames.append(actual_time)

        outfile_displacement.write(solver.displacement_old, time=actual_time)


lim_x, lim_y, lim_z  = list_min_max_coords

interval = 10**3 * output_frequency * sum(time_step_vec)/len(time_step_vec)
animation = animate_vector_triplot(list_frames, interval, \
                                    lim_x = lim_x, \
                                    lim_y = lim_y, \
                                    lim_z = lim_z, three_dim=True)

animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")


n_frames = len(time_frames)
indexes_images = [0, int(n_frames/4), int(n_frames/2), \
                    int(3*n_frames/4), int(n_frames-1)]

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
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel('Time')
plt.legend()
plt.title("Energy")
plt.savefig(f"{directory_results}Energy.pdf", bbox_inches='tight', dpi='figure', format='pdf')

plt.show()