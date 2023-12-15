import firedrake as fdrk
from src.problems.problem import StaticProblem
import matplotlib.pyplot as plt
import time
from firedrake.petsc import PETSc

class NonLinearStaticSolver:
    def __init__(self, problem: StaticProblem, pol_degree=1, formulation="grad"):
        assert formulation=="grad" or formulation=="div"
        
        self.domain = problem.domain
        self.problem = problem

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", pol_degree)
        NED2_vectorspace = fdrk.VectorFunctionSpace(self.domain, "N2curl", pol_degree-1) # Every row is a Nedelec
        BDM_vectorspace = fdrk.VectorFunctionSpace(self.domain, "BDM", pol_degree) # Every row is a BDM
        DG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "DG", pol_degree-1)

        if formulation=="grad":
            self.disp_space = CG_vectorspace
            self.stress_space = NED2_vectorspace
            self.strain_space = NED2_vectorspace
        else:
            self.disp_space = DG_vectorspace
            self.stress_space = BDM_vectorspace
            self.strain_space = BDM_vectorspace

        mixed_space = self.disp_space * self.stress_space * self.strain_space

        test_disp, test_first_piola, test_grad_disp = fdrk.TestFunctions(mixed_space)

        self.solution = fdrk.Function(mixed_space)
        self.displacement, self.first_piola, self.grad_disp = self.solution.subfunctions

        self.delta_solution = fdrk.Function(mixed_space)
        self.delta_displacement, self.delta_first_piola, self.delta_grad_disp = self.delta_solution.subfunctions

        trial_delta_mixed = fdrk.TrialFunction(mixed_space)
        trial_delta_disp, trial_delta_first_piola, trial_delta_grad_disp = fdrk.split(trial_delta_mixed)

        dict_essential_bcs = problem.get_essential_bcs()
        dict_disp_x = dict_essential_bcs["displacement x"]
        dict_disp_y = dict_essential_bcs["displacement y"]

        
        dict_nat_bcs = problem.get_natural_bcs()
        dict_traction_x = dict_nat_bcs["traction x"]
        dict_traction_y = dict_nat_bcs["traction y"]


        if formulation=="grad":
            bcs = []
            for subdomain, disp_x in  dict_disp_x.items():
                bcs.append(fdrk.DirichletBC(mixed_space.sub(0).sub(0), disp_x, subdomain))

            for subdomain, disp_y in  dict_disp_y.items():
                bcs.append(fdrk.DirichletBC(mixed_space.sub(0).sub(1), disp_y, subdomain))

            res_equilibrium = -fdrk.inner(fdrk.grad(test_disp), self.first_piola) * fdrk.dx 

            for subdomain, force_x in dict_traction_x.items():
                res_equilibrium+= fdrk.inner(test_disp[0], force_x) * fdrk.ds(subdomain)

            for subdomain, force_y in dict_traction_y.items():
                res_equilibrium+= fdrk.inner(test_disp[1], force_y) * fdrk.ds(subdomain)

            res_def_grad = fdrk.inner(test_grad_disp, self.grad_disp - fdrk.grad(self.displacement))*fdrk.dx
        
            D_res_u_DP = -fdrk.inner(fdrk.grad(test_disp), trial_delta_first_piola) * fdrk.dx 
            D_res_H_Du = fdrk.inner(test_grad_disp, - fdrk.grad(trial_delta_disp))*fdrk.dx

        else:
            
            bcs = []
            normals = {1: fdrk.Constant((-1, 0)), 2: fdrk.Constant((1, 0)), \
                       3: fdrk.Constant((0, -1)), 4: fdrk.Constant((0, 1))}
            for subdomain, force_x in  dict_traction_x.items():
                space_traction = mixed_space.sub(2).sub(0)
                bcs.append(fdrk.DirichletBC(space_traction, force_x*normals[subdomain], subdomain))
            for subdomain, force_y in  dict_traction_y.items():
                space_traction = mixed_space.sub(2).sub(1)
                bcs.append(fdrk.DirichletBC(space_traction, force_y*normals[subdomain], subdomain))


            res_equilibrium = fdrk.inner(test_disp, fdrk.div(self.first_piola)) * fdrk.dx 

            res_def_grad = fdrk.inner(test_grad_disp, self.grad_disp) * fdrk.dx \
                           + fdrk.inner(fdrk.div(test_grad_disp), self.displacement)*fdrk.dx
            
            for subdomain, disp_x in  dict_disp_x.items():
                res_def_grad-= fdrk.dot(test_grad_disp, problem.normal_versor)[0]*disp_x*fdrk.ds(subdomain)
            for subdomain, disp_y in  dict_disp_y.items():
                res_def_grad-= fdrk.dot(test_grad_disp, problem.normal_versor)[1]*disp_y*fdrk.ds(subdomain)

            D_res_u_DP = fdrk.inner(test_disp, fdrk.div(trial_delta_first_piola)) * fdrk.dx 
            D_res_H_Du = fdrk.inner(fdrk.div(test_grad_disp), trial_delta_disp)*fdrk.dx
        
        res_stress = fdrk.inner(test_first_piola, 
                                self.first_piola - problem.first_piola_definition(self.grad_disp)) * fdrk.dx
        
        actual_res = res_equilibrium + res_stress + res_def_grad

        
        # H is the gradient of the displacement
        D_res_P_DP = fdrk.inner(test_first_piola, trial_delta_first_piola) * fdrk.dx
        D_res_P_DH = fdrk.inner(test_first_piola, 
                                - problem.derivative_first_piola(trial_delta_grad_disp, self.grad_disp)) * fdrk.dx
        D_res_H_DH = fdrk.inner(test_grad_disp, trial_delta_grad_disp)*fdrk.dx

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

        fig, axes = plt.subplots()
        int_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh, self.disp_space))        
        fdrk.triplot(int_coordinates, axes=axes)
        plt.show(block=False)
        plt.pause(0.2)

        for ii in range(n_iter_max):
            self.solver.solve()
            self.solution.assign(self.solution + self.delta_solution)
            axes.cla()
            self.plot_displacement(axes)
            plt.draw()
            plt.pause(0.2)

            PETSc.Sys.Print(f"Norm delta disp at iter {ii+1}: {fdrk.norm(self.delta_displacement)}")
            if fdrk.norm(self.delta_solution)<tolerance:
                break

        

    def plot_displacement(self, axes):
        int_displaced_coordinates = fdrk.Mesh(fdrk.interpolate(self.problem.coordinates_mesh \
                                                               + self.displacement, self.disp_space))

        fdrk.triplot(int_displaced_coordinates, axes=axes)