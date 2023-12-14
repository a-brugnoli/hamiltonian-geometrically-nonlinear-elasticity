import firedrake as fdrk
from src.problems.inhomogeneous_compression import InhomogeneousCompression
from src.solvers.nonlinear_static_grad import NonLinearStaticSolverGrad

nx = 30
ny = 30
pol_degree = 2

problem = InhomogeneousCompression(nx, ny)

solver = NonLinearStaticSolverGrad(problem, pol_degree)

solver.solve()

solver.plot_displacement()