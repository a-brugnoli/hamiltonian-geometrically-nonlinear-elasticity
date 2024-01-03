import firedrake as fdrk
from src.problems.inhomogeneous_compression import InhomogeneousCompression
from src.solvers.nonlinear_static_grad import NonLinearStaticSolver
import numpy as np

nx = 60
ny = 30
pol_degree = 2

# domain = fdrk.RectangleMesh(nx, ny, Lx = 1, Ly = 1)
# BDM_space = fdrk.FunctionSpace(domain, "BDM", pol_degree) # Every row is a BDM

# bc = fdrk.DirichletBC(BDM_space, fdrk.Constant(5)*fdrk.FacetNormal(domain), 1)

problem = InhomogeneousCompression(nx, ny)

solver = NonLinearStaticSolver(problem, pol_degree, formulation="grad")

solver.solve()

