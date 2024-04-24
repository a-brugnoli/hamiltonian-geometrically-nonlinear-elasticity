import firedrake as fdrk
from src.problems.problem import StaticProblem
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from firedrake.exceptions import ConvergenceError
from math import floor

class NonLinearStatic:
    def __init__(self, problem: StaticProblem, num_steps):
        
        self.domain = problem.domain
        self.problem = problem
        self.num_steps = num_steps
        self.loading_factor = fdrk.Constant(0)


    def solve(self, bending=False):

        
        fig, axes = plt.subplots()
        int_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh, self.disp_space))        
        fdrk.triplot(int_coordinates, axes=axes)

        if bending:
            fig_bend = plt.figure()
            axes_bend = fig_bend.add_subplot(111) #, projection='3d')
            axes_bend.set_aspect('equal')

            self.plot_bend_displacement(axes_bend)

        plt.show(block=False)
        plt.pause(0.01)

        not_converged = True
        step = 0
        iteration = 0

        while not_converged:

            previous_solution = self.solution.copy(deepcopy=True)

            iteration+=1
            step +=1
            self.loading_factor.assign(step/self.num_steps)
            PETSc.Sys.Print("Iteration number = %d, step number = %d load factor is %.1f%%" % (iteration, step, self.loading_factor*100))

            try:
                self.solver.solve()

                if step >= self.num_steps - 1:
                    not_converged = False
                    PETSc.Sys.Print(f"Solution converged with {self.num_steps} steps")


            except ConvergenceError:
     
                last_converged_loading_factor = (step - 1)/self.num_steps 
                self.num_steps += 5

                PETSc.Sys.Print(f"Non linear solver fails to converge increasing \
                                number of steps to {self.num_steps}")
               
                step = floor(last_converged_loading_factor * self.num_steps)
                self.solution.assign(previous_solution)
            
            axes.cla()
            self.plot_displacement(axes)

            if bending:
                axes_bend.cla()
                # axes_bend.clear()
                self.plot_bend_displacement(axes_bend)


            plt.draw()
            plt.pause(0.01)

        plt.show()

        
    # def solve(self):

    #     fig, axes = plt.subplots()
    #     int_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh, self.disp_space))        
    #     fdrk.triplot(int_coordinates, axes=axes)
    #     plt.show(block=False)
    #     plt.pause(0.01)

    #     for step in range(self.num_steps):
    #         self.loading_factor.assign((step+1)/self.num_steps)
    #         PETSc.Sys.Print("step number = %d: load factor is %.0f%%" % (step, self.loading_factor*100))

    #         self.solver.solve()

    #         axes.cla()
    #         self.plot_displacement(axes)
    #         plt.draw()
    #         plt.pause(0.01)

    #     plt.show()


    def plot_displacement(self, axes):
        int_displaced_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh \
                                                               + self.displacement, self.disp_space))

        fdrk.triplot(int_displaced_coordinates, axes=axes)


    def plot_bend_displacement(self, axes):
        
        # fdrk.trisurf(fdrk.interpolate(self.bend_displacement, self.bend_disp_space), axes=axes)
        fdrk.tricontourf(fdrk.interpolate(self.bend_displacement, self.bend_disp_space), axes=axes)