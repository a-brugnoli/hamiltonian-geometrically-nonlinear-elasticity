import firedrake as fdrk
from src.problems.problem import StaticProblem
from src.solvers.statics.nonlinear_static import NonLinearStatic
from src.tools.von_karman import membrane_stiffness, bending_compliance, sym_grad

def bilinear_form(testfunctions, functions, normal, parameters):
    
    psi_u, psi_w, psi_M = testfunctions
    u, w, M = functions

    form_laplacian_membrane = fdrk.inner(sym_grad(psi_u), \
        membrane_stiffness(sym_grad(u), parameters)) * fdrk.dx

    membrane_bending_coupling = fdrk.inner(sym_grad(psi_u), membrane_stiffness(0.5*fdrk.outer(fdrk.grad(w), fdrk.grad(w)), parameters)) * fdrk.dx \
        - fdrk.inner(membrane_stiffness(0.5*fdrk.outer(fdrk.grad(psi_w), fdrk.grad(w)), parameters), sym_grad(u)) * fdrk.dx 
            
    nonlinear_laplacian_bending = fdrk.inner(fdrk.outer(fdrk.grad(psi_w), fdrk.grad(w)), 0.5 * fdrk.outer(fdrk.grad(w), fdrk.grad(w))) * fdrk.dx 

    hhj_operator = + fdrk.inner(fdrk.grad(fdrk.grad(psi_w)), M) * fdrk.dx \
    - fdrk.jump(fdrk.grad(psi_w), normal) * fdrk.dot(fdrk.dot(M('+'), normal('+')), normal('+')) * fdrk.dS \
    - fdrk.dot(fdrk.grad(psi_w), normal) * fdrk.dot(fdrk.dot(M, normal), normal) * fdrk.ds \
    - fdrk.inner(psi_M, fdrk.grad(fdrk.grad(w))) * fdrk.dx \
    + fdrk.dot(fdrk.dot(psi_M('+'), normal('+')), normal('+')) * fdrk.jump(fdrk.grad(w), normal) * fdrk.dS \
    + fdrk.dot(fdrk.dot(psi_M, normal), normal) * fdrk.dot(fdrk.grad(w), normal) * fdrk.ds
    
    form_bending_compliance = fdrk.inner(psi_M, bending_compliance(M, parameters)) * fdrk.dx

    total_form = form_laplacian_membrane + form_bending_compliance \
            + membrane_bending_coupling + nonlinear_laplacian_bending + hhj_operator
    
    return total_form


class NonLinearStaticVonKarman(NonLinearStatic):
    def __init__(self, problem: StaticProblem, pol_degree=1, num_steps=1):
        super().__init__(problem, num_steps)
        cell = self.domain.ufl_cell()

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", pol_degree)
        CG_space = fdrk.FunctionSpace(self.domain, "CG", pol_degree)
        HHJ_tensorspace = fdrk.FunctionSpace(self.domain, "HHJ", pol_degree - 1)

        self.mixed_space = CG_vectorspace * CG_space * HHJ_tensorspace

        self.disp_space = CG_vectorspace
        self.bend_disp_space = CG_space

        testfunctions = fdrk.TestFunctions(self.mixed_space)
        test_mem_displacement, test_bend_displacement, test_bend_stress = testfunctions

        self.solution = fdrk.Function(self.mixed_space)
        solution_functions = fdrk.split(self.solution)

        self.displacement, self.bend_displacement, self.bend_moment = solution_functions

        dict_initial_conditions = problem.get_initial_conditions()

        mem_displacement_0 = fdrk.interpolate(dict_initial_conditions["membrane displacement"], CG_vectorspace)
        bend_displacement_0 = fdrk.interpolate(dict_initial_conditions["bending displacement"], CG_space)
        bend_stress_0 = fdrk.interpolate(dict_initial_conditions["bending stress"], HHJ_tensorspace)

        self.solution.sub(0).assign(mem_displacement_0)
        self.solution.sub(1).assign(bend_displacement_0)
        self.solution.sub(2).assign(bend_stress_0)

        dict_essential_bcs = problem.get_essential_bcs()
        dict_disp_x = dict_essential_bcs["displacement x"]
        dict_disp_y = dict_essential_bcs["displacement y"]
        dict_disp_z = dict_essential_bcs["displacement z"]

        dict_bending_moment = dict_essential_bcs["bending stress"]

        bcs = []
        for subdomain, disp_x in  dict_disp_x.items():
            bcs.append(fdrk.DirichletBC(self.mixed_space.sub(0).sub(0), \
                                        self.loading_factor * disp_x, subdomain))

        for subdomain, disp_y in  dict_disp_y.items():
            bcs.append(fdrk.DirichletBC(self.mixed_space.sub(0).sub(1), \
                                        self.loading_factor * disp_y, subdomain))

        for subdomain, disp_z in  dict_disp_z.items():
            bcs.append(fdrk.DirichletBC(self.mixed_space.sub(1), \
                                        self.loading_factor * disp_z, subdomain))

        for subdomain, bend_moment in  dict_bending_moment.items():
            bcs.append(fdrk.DirichletBC(self.mixed_space.sub(2), \
                                        self.loading_factor * bend_moment, subdomain))

        res_equilibrium = bilinear_form(testfunctions, solution_functions,\
                                        problem.normal_versor, problem.parameters)

        forcing = self.problem.get_forcing()

        if forcing is not None:
            res_equilibrium -= fdrk.inner(test_mem_displacement,  self.loading_factor * forcing[0,1]) * fdrk.dx
            res_equilibrium -= fdrk.inner(test_bend_displacement, self.loading_factor * forcing[2]) * fdrk.dx

        dict_nat_bcs = problem.get_natural_bcs()
        dict_traction_x = dict_nat_bcs["traction x"]
        dict_traction_y = dict_nat_bcs["traction y"]

        for subdomain, force_x in dict_traction_x.items():
            res_equilibrium -= self.loading_factor * fdrk.inner(test_mem_displacement[0], 
                                fdrk.dot(force_x, problem.normal_versor)) * fdrk.ds(subdomain)

        for subdomain, force_y in dict_traction_y.items():
            res_equilibrium -= self.loading_factor * fdrk.inner(test_mem_displacement[1], 
                                fdrk.dot(force_y, problem.normal_versor)) * fdrk.ds(subdomain)

        variational_problem = fdrk.NonlinearVariationalProblem(res_equilibrium, 
                                                               self.solution, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)



    