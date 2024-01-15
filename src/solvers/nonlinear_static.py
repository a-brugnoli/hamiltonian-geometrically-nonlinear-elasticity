import firedrake as fdrk
from src.problems.problem import StaticProblem
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
import numpy as np

class NonLinearStaticSolver:
    def __init__(self, problem: StaticProblem, pol_degree=1, formulation="grad", num_steps = 35):
        assert formulation=="grad" or formulation=="div"
        
        self.domain = problem.domain
        self.problem = problem
        self.num_steps = num_steps

        if formulation=="grad":
            cell = self.domain.ufl_cell()

            H1_fe = fdrk.FiniteElement("CG", cell, pol_degree)

            if str(cell)=="triangle":
                Hcurl_fe = fdrk.FiniteElement("N2curl", cell, pol_degree-1, variant=f"integral({pol_degree+1})")
            else:
                Hcurl_fe = fdrk.FiniteElement("RTCE", cell, pol_degree, variant=f"integral")
 
            H1_vectorspace = fdrk.VectorFunctionSpace(self.domain, H1_fe)
            Hcurl_vectorspace = fdrk.VectorFunctionSpace(self.domain, Hcurl_fe) # Every row is from the space

            self.disp_space = H1_vectorspace
            self.strain_space = Hcurl_vectorspace
            self.stress_space = Hcurl_vectorspace
        else:
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

        self.loading_factor = fdrk.Constant(0)


        if formulation=="grad":
            bcs = []
            for subdomain, disp_x in  dict_disp_x.items():
                bcs.append(fdrk.DirichletBC(mixed_space.sub(0).sub(0), disp_x, subdomain))

            for subdomain, disp_y in  dict_disp_y.items():
                bcs.append(fdrk.DirichletBC(mixed_space.sub(0).sub(1), disp_y, subdomain))

            res_equilibrium = fdrk.inner(fdrk.grad(test_disp), self.first_piola) * fdrk.dx 

            for subdomain, force_x in dict_traction_x.items():
                res_equilibrium -= self.loading_factor * fdrk.inner(test_disp[0], 
                                fdrk.dot(force_x, problem.normal_versor)) * fdrk.ds(subdomain)

            for subdomain, force_y in dict_traction_y.items():
                res_equilibrium -= self.loading_factor * fdrk.inner(test_disp[1], 
                                fdrk.dot(force_y, problem.normal_versor)) * fdrk.ds(subdomain)
            
            res_def_grad = fdrk.inner(test_grad_disp,  fdrk.grad(self.displacement) - self.grad_disp)*fdrk.dx
        
            D_res_u_DP = fdrk.inner(fdrk.grad(test_disp), trial_delta_first_piola) * fdrk.dx 
            D_res_H_Du = fdrk.inner(test_grad_disp, fdrk.grad(trial_delta_disp))*fdrk.dx

        else:
            
            bcs = []
                      
            for subdomain, force_x in  dict_traction_x.items():
                space_traction = mixed_space.sub(2).sub(0)
                bcs.append(fdrk.DirichletBC(space_traction, self.loading_factor * force_x, subdomain))
            for subdomain, force_y in  dict_traction_y.items():
                space_traction = mixed_space.sub(2).sub(1)
                bcs.append(fdrk.DirichletBC(space_traction, self.loading_factor * force_y, subdomain))

            res_equilibrium = - fdrk.inner(test_disp, fdrk.div(self.first_piola)) * fdrk.dx 

            res_def_grad = - fdrk.inner(test_grad_disp, self.grad_disp) * fdrk.dx \
                           - fdrk.inner(fdrk.div(test_grad_disp), self.displacement)*fdrk.dx
            
            for subdomain, disp_x in  dict_disp_x.items():
                res_def_grad += fdrk.dot(test_grad_disp, problem.normal_versor)[0]*disp_x*fdrk.ds(subdomain)
            for subdomain, disp_y in  dict_disp_y.items():
                res_def_grad += fdrk.dot(test_grad_disp, problem.normal_versor)[1]*disp_y*fdrk.ds(subdomain)

            D_res_u_DP = - fdrk.inner(test_disp, fdrk.div(trial_delta_first_piola)) * fdrk.dx 
            D_res_H_Du = - fdrk.inner(fdrk.div(test_grad_disp), trial_delta_disp)*fdrk.dx

        
        forcing = self.problem.get_forcing()

        if forcing is not None:
            res_equilibrium -= fdrk.inner(test_disp, self.loading_factor * forcing) * fdrk.dx
        
        res_stress = fdrk.inner(test_first_piola, 
                                problem.first_piola_definition(self.grad_disp) - self.first_piola) * fdrk.dx
        
        actual_res = res_equilibrium + res_def_grad + res_stress
        
        # H is the gradient of the displacement
        D_res_P_DP = fdrk.inner(test_first_piola, - trial_delta_first_piola) * fdrk.dx
        D_res_P_DH = fdrk.inner(test_first_piola, 
                                problem.derivative_first_piola(trial_delta_grad_disp, self.grad_disp)) * fdrk.dx
        
        D_res_H_DH = fdrk.inner(test_grad_disp, - trial_delta_grad_disp)*fdrk.dx

        Jacobian = D_res_u_DP \
                + D_res_H_DH \
                + D_res_H_Du \
                + D_res_P_DP \
                + D_res_P_DH 

        # variational_problem = fdrk.LinearVariationalProblem(Jacobian,
        #                                                     -actual_res, 
        #                                                     self.delta_solution, 
        #                                                     bcs = bcs)

        # self.solver = fdrk.LinearVariationalSolver(variational_problem, solver_parameters={})

        variational_problem = fdrk.NonlinearVariationalProblem(actual_res, 
                                                               self.solution, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)


    def solve(self):

        fig, axes = plt.subplots()
        int_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh, self.disp_space))        
        fdrk.triplot(int_coordinates, axes=axes)
        plt.show(block=False)
        plt.pause(0.2)

        tolerance = 1e-9
        n_iter_max = 20
        damping_factor_newton = 1

        for step in range(self.num_steps):
            self.loading_factor.assign((step+1)/self.num_steps)
            PETSc.Sys.Print("step number = %d: load factor is %.0f%%" % (step, self.loading_factor*100))

            self.solver.solve()


        # for load_step in range(self.num_steps):
        #     self.loading_factor.assign((load_step+1)/self.num_steps)
        #     PETSc.Sys.Print("step number = %d: load factor is %.0f%%" % (load_step, self.loading_factor*100))

        #     for iter in range(n_iter_max):
        #         self.solver.solve()
        #         eps = fdrk.norm(self.delta_solution)
        #         self.solution.assign(self.solution + damping_factor_newton * self.delta_solution)
                
        #         PETSc.Sys.Print("iter = %d: the L2 norm of the increment is %8.2E" % (iter, eps))

        #         if eps < tolerance:
        #             PETSc.Sys.Print("Tolerance reached : exiting Newton loop")
        #             break

            axes.cla()
            self.plot_displacement(axes)
            plt.draw()
            plt.pause(0.2)

        plt.show()
        

    def plot_displacement(self, axes):
        int_displaced_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh \
                                                               + self.displacement, self.disp_space))

        fdrk.triplot(int_displaced_coordinates, axes=axes)