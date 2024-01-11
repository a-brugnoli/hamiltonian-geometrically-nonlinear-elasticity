import firedrake as fdrk
from src.problems.inhomogeneous_compression import InhomogeneousCompression
from src.problems.cook_membrane import CookMembrane
from src.problems.convergence_static import ConvergenceStatic
from src.solvers.nonlinear_static_grad import NonLinearStaticSolverGrad
from src.solvers.nonlinear_static_general import NonLinearStaticSolver
from src.solvers.nonlinear_static_standard import NonLinearStaticSolverStandard

pol_degree = 2

nx = 30
ny = 30
problem = InhomogeneousCompression(nx, ny)

# mesh_size = 2
# problem = CookMembrane(mesh_size)

# problem = ConvergenceStatic(20, 20)

# solver = NonLinearStaticSolver(problem, pol_degree, "grad", 60)

solver = NonLinearStaticSolverGrad(problem, pol_degree, 150)

# solver = NonLinearStaticSolverStandard(problem, pol_degree, 150)

solver.solve()

