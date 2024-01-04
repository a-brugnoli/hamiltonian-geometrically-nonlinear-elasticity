import firedrake as fdrk
from src.problems.inhomogeneous_compression import InhomogeneousCompression
from src.problems.cook_membrane import CookMembrane
from src.solvers.nonlinear_static import NonLinearStaticSolver
import numpy as np


pol_degree = 2

# nx = 60
# ny = 30
# problem = InhomogeneousCompression(nx, ny)

mesh_size = 5
problem = CookMembrane(mesh_size)

# solver = NonLinearStaticSolver(problem, pol_degree, formulation="grad")

# solver.solve()

