import firedrake as fdrk
from src.problems.problem import StaticProblem
import matplotlib.pyplot as plt

class NonLinearStaticSolverGrad:
    def __init__(self, problem: StaticProblem, pol_degree=1):
        
        self.domain = problem.domain
        self.problem = problem

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", pol_degree)
        NED2_vectorspace = fdrk.VectorFunctionSpace(self.domain, "N2curl", pol_degree-1) # Every row is a Nedelec
        # NED1_vectorspace = fdrk.VectorFunctionSpace(self.domain, "N1curl", pol_degree) 
        # BDM_vectorspace = fdrk.VectorFunctionSpace(self.domain, "BDM", pol_degree) # Every row is a BDM

        self.disp_space = CG_vectorspace
        self.stress_space = NED2_vectorspace
        self.strain_space = self.stress_space

        mixed_space_grad = self.disp_space * self.stress_space * self.strain_space

        test_disp, test_first_piola, test_grad_disp = fdrk.TestFunctions(mixed_space_grad)

        self.solution = fdrk.Function(mixed_space_grad)
        self.displacement, self.first_piola, self.grad_disp = self.solution.subfunctions

        self.delta_solution = fdrk.Function(mixed_space_grad)
        delta_displacement, delta_first_piola, delta_grad_disp = self.solution.subfunctions

        dict_essential_bcs = problem.get_essential_bcs()
        dict_disp_x = dict_essential_bcs["displacement x"]

        bcs = []
        for subdomain, value in  dict_disp_x.items():
            bcs.append(fdrk.DirichletBC(mixed_space_grad.sub(0).sub(0), value, subdomain))

        dict_disp_y = dict_essential_bcs["displacement y"]

        for subdomain, value in  dict_disp_y.items():
            bcs.append(fdrk.DirichletBC(mixed_space_grad.sub(0).sub(1), value, subdomain))

        dict_nat_bcs = problem.get_natural_bcs()

        res_equilibrium = fdrk.inner(fdrk.grad(test_disp), self.first_piola) * fdrk.dx 

        for subdomain, force in dict_nat_bcs.items():
            res_equilibrium-= fdrk.inner(test_disp, force) * fdrk.ds(subdomain)

        res_stress = fdrk.inner(test_first_piola, 
                                self.first_piola - problem.first_piola_definition(self.grad_disp)) * fdrk.dx
        res_def_grad = fdrk.inner(test_grad_disp, self.grad_disp - fdrk.grad(self.displacement))*fdrk.dx
        
        actual_res = res_equilibrium + res_stress + res_def_grad

        trial_delta_mixed = fdrk.TrialFunction(mixed_space_grad)
        trial_delta_disp, trial_delta_first_piola, trial_delta_grad_disp = fdrk.split(trial_delta_mixed)

        # H is the gradient of the displacement
        D_res_u_DP = fdrk.inner(fdrk.grad(test_disp), trial_delta_first_piola) * fdrk.dx 
        D_res_P_DP = fdrk.inner(test_first_piola, trial_delta_first_piola) * fdrk.dx
        D_res_P_DH = fdrk.inner(test_first_piola, 
                                - problem.derivative_first_piola(trial_delta_grad_disp, self.grad_disp)) * fdrk.dx
        D_res_H_DH = fdrk.inner(test_grad_disp, trial_delta_grad_disp)*fdrk.dx
        D_res_H_Du = fdrk.inner(test_grad_disp, - fdrk.grad(trial_delta_disp))*fdrk.dx

        Jacobian = D_res_u_DP \
                + D_res_P_DP \
                + D_res_P_DH \
                + D_res_H_DH \
                + D_res_H_Du

        
        variational_problem = fdrk.LinearVariationalProblem(Jacobian,
                                                            -actual_res, 
                                                            self.delta_solution, 
                                                            bcs = bcs)


        self.solver = fdrk.LinearVariationalSolver(variational_problem, solver_parameters={})


    def solve(self):

        tolerance = 1e-9
        n_iter_max = 1000

        for ii in range(n_iter_max):
            print(f"n iter {ii}")
            self.solver.solve()
            self.solution.assign(self.solution + self.delta_solution)

            if fdrk.norm(self.delta_solution)<tolerance:
                break

        

    def plot_displacement(self):


        int_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh, self.disp_space))

        int_displaced_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh \
                                                               + self.displacement, self.disp_space))

        fig, axes = plt.subplots()
        # fdrk.triplot(int_coordinates, axes=axes)
        fdrk.triplot(int_displaced_coordinates, axes=axes)

        plt.show()