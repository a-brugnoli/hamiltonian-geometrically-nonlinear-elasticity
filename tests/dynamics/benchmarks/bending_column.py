import firedrake as fdrk
import matplotlib.pyplot as plt
from src.postprocessing.animators import animate_vector_triplot
from src.solvers.dynamics.hamiltonian_neo_hookean import HamiltonianNeoHookeanSolver
from src.solvers.dynamics.hamiltonian_st_venant import HamiltonianSaintVenantSolver
from src.solvers.dynamics.hamiltonian_st_venant_static_condensation\
    import HamiltonianSaintVenantSolverStaticCondensation

from src.solvers.dynamics.nonlinear_lagrangian_explicit import NonlinearLagrangianExplicitSolver
from src.solvers.dynamics.nonlinear_lagrangian_stormer_verlet\
      import NonlinearLagrangianStormerVerletSolver
from src.problems.dynamics.twisting_column import TwistingColumn
from src.problems.dynamics.bending_column import BendingColumn
import os

from src.tools.elasticity import integrate

pol_degree = 1
T_end = 0.5

problem = BendingColumn(n_elem_x=3, n_elem_y=3, n_elem_z=18)
# problem = BendingColumn(n_elem_x=6, n_elem_y=6, n_elem_z=36)
# problem = BendingColumn(n_elem_x=12, n_elem_y=12, n_elem_z=72)

# solver = HamiltonianSaintVenantSolver(problem, 
#                                     pol_degree)

# solver = HamiltonianSaintVenantSolverStaticCondensation(problem, 
#                                     pol_degree)

# solver = NonlinearLagrangianExplicitSolver(problem, 
#                                         pol_degree, 
#                                         coeff_cfl=0.15)

solver = NonlinearLagrangianStormerVerletSolver(problem, 
                                        pol_degree, 
                                        coeff_cfl=0.2)

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)

home_dir =os.environ['HOME']
directory_largedata = f"{home_dir}/StoreResults/{str(solver)}/{str(problem)}/"
if not os.path.exists(directory_largedata):
    os.makedirs(directory_largedata, exist_ok=True)

outfile_displacement = fdrk.File(f"{directory_largedata}/Displacement.pvd")

outfile_displacement_condensation = fdrk.File(f"{directory_largedata}/Displacement.pvd")

output_frequency = 10

dict_results = integrate(solver, T_end, \
                        outfile_displacement, \
                        output_frequency)

time_vector = dict_results['time']
energy_vector = dict_results['energy']
list_displacement = dict_results["displacement solution"]
computing_time = dict_results["computing time"]
list_frames =  dict_results["displacement mesh"]
list_min_max_coords = dict_results["minmax displacement"]
time_step_vec = dict_results["time steps"]

lim_x, lim_y, lim_z  = list_min_max_coords

interval = 10**3 * output_frequency * sum(time_step_vec)/len(time_step_vec)
animation = animate_vector_triplot(list_frames, interval, \
                                    lim_x = lim_x, \
                                    lim_y = lim_y, \
                                    lim_z = lim_z, three_dim=True)

# animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")

# n_frames = len(time_frames)
# indexes_images = [0, int(n_frames/4), int(n_frames/2), \
#                     int(3*n_frames/4), int(n_frames-1)]

# for kk in indexes_images:
#     time_image = time_frames[kk]

#     fig = plt.figure()
#     axes = fig.add_subplot(111, projection='3d')
#     axes.set_aspect('equal')
#     fdrk.triplot(list_frames[kk], axes=axes)
#     axes.set_title(f"Displacement $t={time_image:.1f}$ [ms]", loc='center')
#     axes.set_xlabel("x")
#     axes.set_ylabel("y")
#     axes.set_xlim(lim_x)
#     axes.set_ylim(lim_y)
#     axes.set_zlim(lim_z)

#     plt.savefig(f"{directory_results}/Displacement_t{time_image:.1f}.pdf", bbox_inches='tight', dpi='figure', format='pdf')


plt.figure()
plt.plot(time_vector, energy_vector)
plt.grid(color='0.8', linestyle='-', linewidth=.5)
plt.xlabel('Time $\mathrm[s]$')
plt.legend()
plt.title("Energy")
plt.savefig(f"{directory_results}Energy.pdf", bbox_inches='tight', dpi='figure', format='pdf')

plt.show()


