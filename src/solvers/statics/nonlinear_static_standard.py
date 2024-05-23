import firedrake as fdrk
from src.problems.problem import StaticProblem
from src.solvers.statics.nonlinear_static import NonLinearStatic
from src.tools.elasticity import first_piola_neohookean

class NonLinearStaticSolverStandard(NonLinearStatic):
    def __init__(self, problem: StaticProblem, pol_degree=2, num_steps=1):
        super().__init__(problem, num_steps)
        
        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", pol_degree)
        self.disp_space = CG_vectorspace
        test_disp = fdrk.TestFunction(self.disp_space)

        self.solution = fdrk.Function(self.disp_space)
        self.displacement = self.solution
        self.solution.assign(fdrk.as_vector([0] * problem.dim))
        
        self.delta_disp = fdrk.Function(self.disp_space)

        dict_essential_bcs = problem.get_essential_bcs()
        dict_disp_x = dict_essential_bcs["displacement x"]
        dict_disp_y = dict_essential_bcs["displacement y"]

        bcs = []
        for subdomain, disp_x in  dict_disp_x.items():
            bcs.append(fdrk.DirichletBC(self.disp_space.sub(0), disp_x, subdomain))

        for subdomain, disp_y in  dict_disp_y.items():
            bcs.append(fdrk.DirichletBC(self.disp_space.sub(1), disp_y, subdomain))

        dict_nat_bcs = problem.get_natural_bcs()
        dict_traction_x = dict_nat_bcs["traction x"]
        dict_traction_y = dict_nat_bcs["traction y"]

        first_piola = first_piola_neohookean(fdrk.grad(self.solution), problem.parameters)

        res_equilibrium = fdrk.inner(fdrk.grad(test_disp), first_piola) * fdrk.dx 

        for subdomain, force_x in dict_traction_x.items():
            res_equilibrium -= self.loading_factor * fdrk.inner(test_disp[0], 
                                fdrk.dot(force_x, problem.normal_versor)) * fdrk.ds(subdomain)

        for subdomain, force_y in dict_traction_y.items():
            res_equilibrium -= self.loading_factor * fdrk.inner(test_disp[1], 
                                fdrk.dot(force_y, problem.normal_versor)) * fdrk.ds(subdomain)

        forcing = self.problem.get_forcing()

        if forcing is not None:
            res_equilibrium -= fdrk.inner(test_disp, self.loading_factor * forcing) * fdrk.dx

        variational_problem = fdrk.NonlinearVariationalProblem(res_equilibrium, 
                                                               self.solution, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)

