import firedrake as fdrk
from src.problems.problem import StaticProblem
import matplotlib.pyplot as plt
import time
from firedrake.petsc import PETSc

class NonLinearStaticSolverStandard:
    def __init__(self, problem: StaticProblem, pol_degree=2, num_steps=35):
        
        self.domain = problem.domain
        self.problem = problem
        self.num_steps = num_steps

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", pol_degree)
        
        self.disp_space = CG_vectorspace
        
        test_disp = fdrk.TestFunction(self.disp_space)

        self.displacement = fdrk.Function(self.disp_space)

        self.displacement.assign(fdrk.as_vector([0] * problem.dim))
        
        self.delta_disp = fdrk.Function(self.disp_space)

        trial_delta_disp = fdrk.TrialFunction(self.disp_space)

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

        first_piola = problem.first_piola_definition(fdrk.grad(self.displacement))

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

        # variational_problem = fdrk.LinearVariationalProblem(Jacobian,
        #                                                     -actual_res, 
        #                                                     self.delta_solution, 
        #                                                     bcs = bcs)


        # self.solver = fdrk.LinearVariationalSolver(variational_problem, solver_parameters={})

        variational_problem = fdrk.NonlinearVariationalProblem(res_equilibrium, 
                                                               self.displacement, 
                                                               bcs = bcs)

        self.solver = fdrk.NonlinearVariationalSolver(variational_problem)



    def solve(self):

        fig, axes = plt.subplots()
        int_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh, self.disp_space))        
        fdrk.triplot(int_coordinates, axes=axes)
        plt.show(block=False)
        plt.pause(0.2)

        for step in range(self.num_steps):
            self.loading_factor.assign((step+1)/self.num_steps)
            PETSc.Sys.Print("step number = %d: load factor is %.0f%%" % (step, self.loading_factor*100))

            self.solver.solve()

            axes.cla()
            self.plot_displacement(axes)
            plt.draw()
            plt.pause(0.2)

        plt.show(block=False)


    def plot_displacement(self, axes):
        int_displaced_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh \
                                                               + self.displacement, self.disp_space))

        fdrk.triplot(int_displaced_coordinates, axes=axes)