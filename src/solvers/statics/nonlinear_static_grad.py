import firedrake as fdrk
from src.problems.problem import StaticProblem
from src.solvers.statics.nonlinear_static import NonLinearStatic


class NonLinearStaticSolverGrad(NonLinearStatic):
    def __init__(self, problem: StaticProblem, pol_degree=2, num_steps=35):
        super().__init__(problem, num_steps)
        
        cell = self.domain.ufl_cell()

        H1_fe = fdrk.FiniteElement("CG", cell, pol_degree)

        if str(cell)=="triangle":
            Hcurl_fe = fdrk.FiniteElement("N2curl", cell, pol_degree-1, variant=f"integral({pol_degree+2})")
        else:
            Hcurl_fe = fdrk.FiniteElement("RTCE", cell, pol_degree)
 
        H1_vectorspace = fdrk.VectorFunctionSpace(self.domain, H1_fe)
        Hcurl_vectorspace = fdrk.VectorFunctionSpace(self.domain, Hcurl_fe) # Every row is from the space

        self.disp_space = H1_vectorspace
        self.strain_space = Hcurl_vectorspace
        self.stress_space = Hcurl_vectorspace

        mixed_space = self.disp_space * self.strain_space * self.stress_space

        test_disp, test_grad_disp, test_first_piola = fdrk.TestFunctions(mixed_space)

        self.solution = fdrk.Function(mixed_space)
        self.displacement, self.grad_disp, self.first_piola = fdrk.split(self.solution)

        self.solution.sub(0).assign(fdrk.as_vector([0] * problem.dim))
        self.solution.sub(1).assign(fdrk.as_tensor([([0] * problem.dim) for i in range(problem.dim)]))
        self.solution.sub(2).assign(fdrk.as_tensor([([0] * problem.dim) for i in range(problem.dim)]))

        self.delta_solution = fdrk.Function(mixed_space)
        self.delta_displacement, self.delta_grad_disp, self.delta_first_piola = self.delta_solution.subfunctions

        trial_delta_mixed = fdrk.TrialFunction(mixed_space)
        # trial_delta_disp, trial_delta_grad_disp, trial_delta_first_piola = fdrk.split(trial_delta_mixed)

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

        res_equilibrium = fdrk.inner(fdrk.grad(test_disp), self.first_piola) * fdrk.dx 

        for subdomain, force_x in dict_traction_x.items():
            res_equilibrium -= self.loading_factor * fdrk.inner(test_disp[0], 
                                fdrk.dot(force_x, problem.normal_versor)) * fdrk.ds(subdomain)

        for subdomain, force_y in dict_traction_y.items():
            res_equilibrium -= self.loading_factor * fdrk.inner(test_disp[1], 
                                fdrk.dot(force_y, problem.normal_versor)) * fdrk.ds(subdomain)

        forcing = self.problem.get_forcing()

        if forcing is not None:
            res_equilibrium -= fdrk.inner(test_disp, self.loading_factor * forcing) * fdrk.dx

        res_def_grad = fdrk.inner(test_grad_disp,  fdrk.grad(self.displacement) - self.grad_disp)*fdrk.dx
        
        res_stress = fdrk.inner(test_first_piola, 
                                problem.first_piola_definition(self.grad_disp) - self.first_piola) * fdrk.dx
        
        actual_res = res_equilibrium + res_def_grad + res_stress

        # Linearization of the residual to use a Newton method
        # D_res_u_DP = fdrk.inner(fdrk.grad(test_disp), trial_delta_first_piola) * fdrk.dx 

        # D_res_H_Du = fdrk.inner(test_grad_disp, fdrk.grad(trial_delta_disp))*fdrk.dx
        # D_res_H_DH = fdrk.inner(test_grad_disp, - trial_delta_grad_disp)*fdrk.dx

        # D_res_P_DH = fdrk.inner(test_first_piola, 
        #                         problem.derivative_first_piola(trial_delta_grad_disp, self.grad_disp)) * fdrk.dx

        # D_res_P_DP = fdrk.inner(test_first_piola, - trial_delta_first_piola) * fdrk.dx

        # Jacobian = D_res_u_DP \
        #             + D_res_H_DH \
        #             + D_res_H_Du \
        #             + D_res_P_DP \
        #             + D_res_P_DH

        variational_problem = fdrk.NonlinearVariationalProblem(actual_res, 
                                                               self.solution, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)

