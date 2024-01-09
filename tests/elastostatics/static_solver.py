import firedrake as fdrk
from src.problems.inhomogeneous_compression import InhomogeneousCompression
from src.problems.cook_membrane import CookMembrane
from src.problems.convergence_static import ConvergenceStatic
from src.solvers.nonlinear_static_grad import NonLinearStaticSolverGrad
from src.solvers.nonlinear_static_general import NonLinearStaticSolver
from src.solvers.nonlinear_static_standard import NonLinearStaticSolverStandard

import numpy as np

pol_degree = 3

# nx = 30
# ny = 30
# problem = InhomogeneousCompression(nx, ny)

mesh_size = 1
problem = CookMembrane(mesh_size)

# problem = ConvergenceStatic(20, 20)

# solver = NonLinearStaticSolver(problem, pol_degree, formulation="grad")
# solver = NonLinearStaticSolverGrad(problem, pol_degree)

solver = NonLinearStaticSolverStandard(problem, pol_degree)

solver.solve()

import matplotlib.pyplot as plt
fig, axes = plt.subplots()
solver.plot_displacement(axes)
plt.show()