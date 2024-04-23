import firedrake as fdrk
from src.problems.problem import StaticProblem
from src.solvers.statics.nonlinear_static import NonLinearStatic
from src.tools.von_karman import membrane_stiffness, bending_compliance, sym_grad

def bilinear_form(testfunctions, functions, normal):
    
    psi_u, psi_w, psi_M = testfunctions
    u, w, M = functions

    form_laplacian_membrane = fdrk.inner(sym_grad(psi_u), membrane_stiffness(sym_grad(w))) * fdrk.dx

    membrane_bending_coupling = fdrk.inner(sym_grad(psi_u)), membrane_stiffness(0.5*fdrk.outer(fdrk.grad(w), fdrk.grad(w))) * fdrk.dx \
        - fdrk.inner(membrane_stiffness(0.5*fdrk.outer(fdrk.grad(psi_w), fdrk.grad(w)), sym_grad(u))) * fdrk.dx 
            
    nonlinear_laplacian_bending = fdrk.inner(fdrk.outer(fdrk.grad(psi_w), fdrk.grad(w)), 0.5 * fdrk.outer(fdrk.grad(w), fdrk.grad(w))) * fdrk.dx 

    hhj_operator = + fdrk.inner(fdrk.grad(fdrk.grad(psi_w)), M) * fdrk.dx \
    - fdrk.jump(fdrk.grad(psi_w), normal) * fdrk.dot(fdrk.dot(M('+'), normal('+')), normal('+')) * fdrk.dS \
    - fdrk.dot(fdrk.grad(psi_w), normal) * fdrk.dot(fdrk.dot(M, normal), normal) * fdrk.ds \
    - fdrk.inner(psi_M, fdrk.grad(fdrk.grad(w))) * fdrk.dx \
    + fdrk.dot(fdrk.dot(psi_M('+'), normal('+')), normal('+')) * fdrk.jump(fdrk.grad(w), normal) * fdrk.dS \
    + fdrk.dot(fdrk.dot(psi_M, normal), normal) * fdrk.dot(fdrk.grad(w), normal) * fdrk.ds
    
    form_bending_compliance = fdrk.inner(psi_M, bending_compliance(M)) * fdrk.dx

    total_form = form_laplacian_membrane + form_bending_compliance \
            + membrane_bending_coupling + nonlinear_laplacian_bending + hhj_operator
    
    return total_form


class NonLinearStaticVonKarman(NonLinearStatic):
    def __init__(self, problem: StaticProblem, pol_degree=1, num_steps=1):
        super().__init__(problem, num_steps)
        cell = self.domain.ufl_cell()

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", pol_degree)
        CG_space = fdrk.FunctionSpace(self.domain, "CG", pol_degree)
        HHJ_tensorspace = fdrk.FunctionSpace(self.domain, "CG", pol_degree - 1)

        self.mixed_space = CG_vectorspace * CG_space * HHJ_tensorspace

        self.disp_space = CG_vectorspace

        testfunctions = fdrk.TestFunctions(self.mixed_space)
        test_mem_displacement, test_bend_displacement, test_bend_stress = testfunctions

        self.solution = fdrk.Function(self.disp_space)
        self.displacement, self.bend_displacement, self.bending_stress = self.solution.subfunctions
        

        dict_essential_bcs = problem.get_essential_bcs()
        dict_disp_x = dict_essential_bcs["displacement x"]
        dict_disp_y = dict_essential_bcs["displacement y"]
        dict_disp_z = dict_essential_bcs["displacement z"]



        bcs = []
        for subdomain, disp_x in  dict_disp_x.items():
            bcs.append(fdrk.DirichletBC(self.mixed_space.sub(0), disp_x, subdomain))

        for subdomain, disp_y in  dict_disp_y.items():
            bcs.append(fdrk.DirichletBC(self.disp_space.sub(1), disp_y, subdomain))

        for subdomain, disp_z in  dict_disp_z.items():
            bcs.append(fdrk.DirichletBC(self.disp_space.sub(2), disp_z, subdomain))

        
        forcing = self.problem.get_forcing()

        res_equilibrium = bilinear_form(testfunctions, self.solution, problem.normal_versor)

        if forcing is not None:
            res_equilibrium -= fdrk.inner(test_mem_displacement,  self.loading_factor * forcing[0,1]) * fdrk.dx
            res_equilibrium -= fdrk.inner(test_bend_displacement, self.loading_factor * forcing[2]) * fdrk.dx

        variational_problem = fdrk.NonlinearVariationalProblem(res_equilibrium, 
                                                               self.solution, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)


