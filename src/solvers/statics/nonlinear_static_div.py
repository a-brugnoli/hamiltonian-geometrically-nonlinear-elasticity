import firedrake as fdrk
from src.problems.problem import StaticProblem
from src.solvers.statics.nonlinear_static import NonLinearStatic

class NonLinearStaticSolverDiv(NonLinearStatic):
    def __init__(self, problem: StaticProblem, pol_degree=1, num_steps = 1):
        super().__init__(problem, num_steps)

        # self.domain = problem.domain
        # self.problem = problem
        # self.num_steps = num_steps
        # self.loading_factor = fdrk.Constant(0)

        cell = self.domain.ufl_cell()
        L2_fe = fdrk.FiniteElement("DG", cell, pol_degree-1)

        if str(cell)=="triangle":
            Hdiv_fe = fdrk.FiniteElement("BDM", cell, pol_degree, 
                                        variant=f"integral({pol_degree+1})")
        else:
            Hdiv_fe = fdrk.FiniteElement("RTCF", cell, pol_degree)

        L2_vectorspace = fdrk.VectorFunctionSpace(self.domain, L2_fe)
        Hdiv_vectorspace = fdrk.VectorFunctionSpace(self.domain, Hdiv_fe) 

        self.disp_space = L2_vectorspace
        self.strain_space = Hdiv_vectorspace
        self.stress_space = Hdiv_vectorspace

        mixed_space = self.disp_space * self.strain_space * self.stress_space

        test_disp, test_grad_disp, test_first_piola = fdrk.TestFunctions(mixed_space)

        self.solution = fdrk.Function(mixed_space)
        # self.displacement, self.grad_disp, self.first_piola = self.solution.subfunctions
        self.displacement, self.grad_disp, self.first_piola = fdrk.split(self.solution)

        self.solution.sub(0).assign(fdrk.as_vector([0] * problem.dim))
        self.solution.sub(1).assign(fdrk.as_tensor([([0] * problem.dim) for i in range(problem.dim)]))
        self.solution.sub(2).assign(fdrk.as_tensor([([0] * problem.dim) for i in range(problem.dim)]))

        self.delta_solution = fdrk.Function(mixed_space)
        self.delta_displacement, self.delta_grad_disp, self.delta_first_piola = self.delta_solution.subfunctions

        trial_delta_mixed = fdrk.TrialFunction(mixed_space)
        trial_delta_disp, trial_delta_grad_disp, trial_delta_first_piola = fdrk.split(trial_delta_mixed)

        dict_essential_bcs = problem.get_essential_bcs()
        dict_disp_x = dict_essential_bcs["displacement x"]
        dict_disp_y = dict_essential_bcs["displacement y"]

        dict_nat_bcs = problem.get_natural_bcs()
        dict_traction_x = dict_nat_bcs["traction x"]
        dict_traction_y = dict_nat_bcs["traction y"]
            
        bcs = []
                    
        for subdomain, force_x in  dict_traction_x.items():
            space_traction = mixed_space.sub(2).sub(0)
            bcs.append(fdrk.DirichletBC(space_traction, self.loading_factor * force_x, subdomain))
        for subdomain, force_y in  dict_traction_y.items():
            space_traction = mixed_space.sub(2).sub(1)
            bcs.append(fdrk.DirichletBC(space_traction, self.loading_factor * force_y, subdomain))

        res_equilibrium = - fdrk.inner(test_disp, fdrk.div(self.first_piola)) * fdrk.dx 

        forcing = self.problem.get_forcing()

        if forcing is not None:
            res_equilibrium -= fdrk.inner(test_disp, self.loading_factor * forcing) * fdrk.dx
        
        res_def_grad = - fdrk.inner(test_grad_disp, self.grad_disp) * fdrk.dx \
                        - fdrk.inner(fdrk.div(test_grad_disp), self.displacement)*fdrk.dx
        
        for subdomain, disp_x in  dict_disp_x.items():
            res_def_grad += fdrk.dot(test_grad_disp, problem.normal_versor)[0]*disp_x*fdrk.ds(subdomain)
        for subdomain, disp_y in  dict_disp_y.items():
            res_def_grad += fdrk.dot(test_grad_disp, problem.normal_versor)[1]*disp_y*fdrk.ds(subdomain)

        res_stress = fdrk.inner(test_first_piola, 
                                problem.first_piola_definition(self.grad_disp) - self.first_piola) * fdrk.dx
        
        actual_res = res_equilibrium + res_def_grad + res_stress
        
        # # Linearization of the residual for use in a Newton method
        # D_res_u_DP = - fdrk.inner(test_disp, fdrk.div(trial_delta_first_piola)) * fdrk.dx 
        # D_res_H_Du = - fdrk.inner(fdrk.div(test_grad_disp), trial_delta_disp)*fdrk.dx
        # D_res_H_DH = fdrk.inner(test_grad_disp, - trial_delta_grad_disp)*fdrk.dx
        # D_res_P_DP = fdrk.inner(test_first_piola, - trial_delta_first_piola) * fdrk.dx
        # D_res_P_DH = fdrk.inner(test_first_piola, 
        #                         problem.derivative_first_piola(trial_delta_grad_disp, self.grad_disp)) * fdrk.dx
    
        # Jacobian = D_res_u_DP \
        #         + D_res_H_DH \
        #         + D_res_H_Du \
        #         + D_res_P_DP \
        #         + D_res_P_DH 

        variational_problem = fdrk.NonlinearVariationalProblem(actual_res, 
                                                               self.solution, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)
