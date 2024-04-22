from src.preprocessing.basic_plotting import *
from src.problems.inhomogeneous_compression import InhomogeneousCompression
from src.problems.cook_membrane import CookMembrane
from src.problems.convergence_static import ConvergenceStatic
from src.solvers.statics.nonlinear_static_grad import NonLinearStaticSolverGrad
from src.solvers.statics.nonlinear_static_standard import NonLinearStaticSolverStandard
from src.solvers.statics.nonlinear_static_secondpiola import NonLinearStaticSolverGradSecPiola


problem_id = 2
solver_id = 1
pol_degree = 2

match problem_id:
    case 1:
        problem = ConvergenceStatic(20, 20)
        num_steps = 35
    case 2:
        mesh_size = 2
        problem = CookMembrane(mesh_size)
        num_steps = 5
    case 3:
        nx = 30
        ny = 30
        problem = InhomogeneousCompression(nx, ny, quad=True)
        num_steps = 150
    case _:
        print("Invalid problem id") 


match solver_id:
    case 1:
        solver = NonLinearStaticSolverStandard(problem, pol_degree, num_steps)
    case 2:
        solver = NonLinearStaticSolverGrad(problem, pol_degree, num_steps)   
    case 3:
        solver = NonLinearStaticSolverGradSecPiola(problem, pol_degree, num_steps)
    case _:
        print("Invalid solver id") 

solver.solve()