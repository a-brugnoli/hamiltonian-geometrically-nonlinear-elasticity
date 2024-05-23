import firedrake as fdrk
from src.problems.problem import StaticProblem
from src.solvers.statics.nonlinear_static import NonLinearStatic
from src.tools.elasticity import second_piola_neohookean


class NonLinearStaticSolverGradSecPiola(NonLinearStatic):
    def __init__(self, problem: StaticProblem, pol_degree=2, num_steps=1):
        super().__init__(problem, num_steps)
        
        # self.domain = problem.domain
        # self.problem = problem
        # self.num_steps = num_steps
        # self.loading_factor = fdrk.Constant(0)

        cell = self.domain.ufl_cell()

        assert problem.dim==2

        H1_fe = fdrk.FiniteElement("CG", cell, pol_degree)

        
        H1_vectorspace = fdrk.VectorFunctionSpace(self.domain, H1_fe)
        self.disp_space = H1_vectorspace


        # regge_broken_fe = fdrk.BrokenElement(fdrk.FiniteElement("Regge", cell, pol_degree-1))
        regge_fe = fdrk.FiniteElement("Regge", cell, pol_degree)
        regge_space = fdrk.FunctionSpace(problem.domain, regge_fe)

        DG_fe = fdrk.FiniteElement("DG", cell, pol_degree)
        DG_symtensorspace = fdrk.TensorFunctionSpace(self.domain, DG_fe) 

        self.stress_space = regge_space

        mixed_space = self.disp_space * self.stress_space

        test_disp, test_second_piola = fdrk.TestFunctions(mixed_space)

        self.solution = fdrk.Function(mixed_space)
        self.displacement, self.second_piola = fdrk.split(self.solution)
        
        dict_essential_bcs = problem.get_essential_bcs()
        dict_disp_x = dict_essential_bcs["displacement x"]
        dict_disp_y = dict_essential_bcs["displacement y"]

        bcs = []
        for subdomain, disp_x in  dict_disp_x.items():
            bcs.append(fdrk.DirichletBC(mixed_space.sub(0).sub(0), disp_x, subdomain))

        for subdomain, disp_y in  dict_disp_y.items():
            bcs.append(fdrk.DirichletBC(mixed_space.sub(0).sub(1), disp_y, subdomain))

        dict_nat_bcs = problem.get_natural_bcs()
        dict_traction_x = dict_nat_bcs["traction x"]
        dict_traction_y = dict_nat_bcs["traction y"]

        def_gradient = fdrk.Identity(problem.dim) + fdrk.grad(self.displacement)
        first_piola = fdrk.dot(def_gradient, self.second_piola)

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

        cauchy_strain = fdrk.dot(def_gradient.T, def_gradient)
        
        res_stress = fdrk.inner(test_second_piola, 
                                second_piola_neohookean(cauchy_strain, problem.parameters) \
                                - self.second_piola) * fdrk.dx
        
        actual_res = res_equilibrium  + res_stress

        variational_problem = fdrk.NonlinearVariationalProblem(actual_res, 
                                                               self.solution, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)



        