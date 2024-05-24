import firedrake as fdrk
import matplotlib.pyplot as plt
from src.postprocessing.animators import animate_vector_triplot
from src.solvers.dynamics.hamiltonian_neo_hookean import HamiltonianNeoHookeanSolver
from src.solvers.dynamics.hamiltonian import HamiltonianSaintVenantSolver
from src.solvers.dynamics.hamiltonian_static_condensation\
    import HamiltonianSaintVenantSolverStaticCondensation

from src.solvers.dynamics.nonlinear_explicit_newmark import NonlineaExplicitNewmarkSolver
from src.solvers.dynamics.nonlinear_dual_stormer_verlet\
      import NonlinearDualStormerVerletSolver
from src.solvers.dynamics.nonlinear_stormer_verlet\
      import NonlinearStormerVerletSolver

from src.problems.dynamics.twisting_column import TwistingColumn
from src.problems.dynamics.bending_column import BendingColumn
import os
from src.tools.elasticity import integrate
import numpy as np

save_figs = True
pol_degree = 1
T_end = 2

problem = BendingColumn(n_elem_x=3, n_elem_y=3, n_elem_z=18)
# problem = BendingColumn(n_elem_x=6, n_elem_y=6, n_elem_z=36)
# problem = BendingColumn(n_elem_x=12, n_elem_y=12, n_elem_z=72)

# solver = HamiltonianSaintVenantSolver(problem, 
#                                     pol_degree)

# solver_energy_conserving = HamiltonianSaintVenantSolverStaticCondensation(problem, 
#                                     pol_degree,
#                                     coeff_cfl=0.8)

solver_stormer_verlet = NonlinearStormerVerletSolver(problem, 
                                            pol_degree, 
                                            coeff_cfl=0.18)

# solver_stormer_verlet = NonlinearDualStormerVerletSolver(problem, 
#                                             pol_degree, 
#                                             coeff_cfl=0.18)


# directory_results = f"{os.path.dirname(os.path.abspath(__file__))}\
#                     /results/{str(solver_stormer_verlet)}/{str(problem)}/"
# if not os.path.exists(directory_results):
#     os.makedirs(directory_results)

output_frequency = 10

dict_results_sv = integrate(solver_stormer_verlet, T_end, \
                            output_frequency = output_frequency, \
                            collect_frames=False)

# dict_results_ec = integrate(solver_energy_conserving, T_end, \
#                             output_frequency = output_frequency)

time_vector_sv = dict_results_sv['time']
energy_vector_sv = dict_results_sv['energy']
computing_time_sv = dict_results_sv["computing time"]
computing_time_sv = dict_results_sv["computing time"]

# time_vector_ec = dict_results_ec['time']
# energy_vector_ec = dict_results_ec['energy']
# computing_time_ec = dict_results_ec["computing time"]

# print(f"Computing time Stormer Verlet : {computing_time_sv} (s)")
# print(f"Computing time Energy Coonserving : {computing_time_ec} (s)")

# plt.figure()
# # plt.plot(time_vector_sv, energy_vector_sv, label="Stormer Verlet")
# plt.plot(time_vector_ec, energy_vector_ec, label="Energy Conserving")
# plt.grid(color='0.8', linestyle='-', linewidth=.5)
# plt.xlabel('Time $\mathrm[s]$')
# plt.legend()
# plt.title("Energy")

# if save_figs:
#     plt.savefig(f"{directory_results}Energy.pdf", bbox_inches='tight', dpi='figure', format='pdf')

# plt.show()



# lim_x, lim_y, lim_z  = list_min_max_coords

# interval = 10**3 * output_frequency * sum(time_step_vec)/len(time_step_vec)

# animation = animate_vector_triplot(list_frames, interval, \
#                                     lim_x = lim_x, \
#                                     lim_y = lim_y, \
#                                     lim_z = lim_z, three_dim=True)

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
