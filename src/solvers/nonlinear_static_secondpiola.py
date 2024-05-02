import firedrake as fdrk
from src.problems.problem import StaticProblem
import matplotlib.pyplot as plt
import time
from firedrake.petsc import PETSc

class NonLinearStaticSolverGradSecPiola:
    def __init__(self, problem: StaticProblem, pol_degree=2, num_steps=35):
        
        self.domain = problem.domain
        self.problem = problem
        self.num_steps = num_steps

        cell = self.domain.ufl_cell()

        H1_fe = fdrk.FiniteElement("CG", cell, pol_degree)
        L2_fe = fdrk.FiniteElement("DG", cell, pol_degree-1)

        H1_vectorspace = fdrk.VectorFunctionSpace(self.domain, H1_fe)
        L2_symtensorspace = fdrk.TensorFunctionSpace(self.domain, L2_fe) 

        self.disp_space = H1_vectorspace
        self.stress_space = L2_symtensorspace

        mixed_space = self.disp_space * self.stress_space

        test_disp, test_second_piola = fdrk.TestFunctions(mixed_space)

        self.solution = fdrk.Function(mixed_space)
        self.displacement, self.second_piola = fdrk.split(self.solution)

        self.solution.sub(0).assign(fdrk.as_vector([0] * problem.dim))
        self.solution.sub(1).assign(fdrk.as_tensor([([0] * problem.dim) for i in range(problem.dim)]))

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

        self.loading_factor = fdrk.Constant(0)

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
                                problem.second_piola_definition(cauchy_strain) - self.second_piola) * fdrk.dx
        
        actual_res = res_equilibrium  + res_stress

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