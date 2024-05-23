import firedrake as fdrk
from src.problems.problem import DynamicProblem
import numpy as np
from src.tools.elasticity import stiffness_tensor, def_gradient, \
    green_lagrange_strain, natural_control_follower
from src.tools.common import compute_time_step

class NonlinearLagrangianStormerVerletSolver:
    def __init__(self,
                 problem: DynamicProblem,
                 pol_degree= 1,
                 coeff_cfl = 0.8,
                 solver_parameters= {}):
        
        # Set times 
        self.problem = problem
        self.coeff_clf = coeff_cfl

        rho = problem.parameters["rho"]
        E = problem.parameters["E"]
        nu = problem.parameters["nu"]
    
        self.CG_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)

        # Spaces and functions
        self.displacement_old = fdrk.Function(self.CG_vectorspace)
        self.displacement_oldmid = fdrk.Function(self.CG_vectorspace)
        self.displacement_newmid = fdrk.Function(self.CG_vectorspace)
        self.displacement_new = 0.5*(self.displacement_oldmid + self.displacement_newmid)

        self.green_lagrange_strain_oldmid = green_lagrange_strain(self.displacement_oldmid)
        self.second_piola_stress_oldmid = stiffness_tensor(self.green_lagrange_strain_oldmid, E, nu)
        self.first_piola_stress_oldmid = fdrk.dot(def_gradient(self.displacement_oldmid), \
                                        self.second_piola_stress_oldmid)
        
        self.green_lagrange_strain_old  = green_lagrange_strain(self.displacement_old)
        self.second_piola_stress_old = stiffness_tensor(self.green_lagrange_strain_old, E, nu)
        self.first_piola_stress_old = fdrk.dot(def_gradient(self.displacement_old), \
                                        self.second_piola_stress_old)
        
        self.green_lagrange_strain_new = green_lagrange_strain(self.displacement_new)
        self.second_piola_stress_new = stiffness_tensor(self.green_lagrange_strain_new, E, nu)
        self.first_piola_stress_new = fdrk.dot(def_gradient(self.displacement_new), \
                                        self.second_piola_stress_new)
        
        self.velocity_old = fdrk.Function(self.CG_vectorspace)
        self.velocity_new = fdrk.Function(self.CG_vectorspace)

        test_CG = fdrk.TestFunction(self.CG_vectorspace)
        trial_CG = fdrk.TrialFunction(self.CG_vectorspace)
        
        expression_initial = problem.get_initial_conditions()
        displacement_t0 = expression_initial["displacement"]
        velocity_t0 = expression_initial["velocity"]

        self.displacement_old.assign(fdrk.interpolate(displacement_t0, self.CG_vectorspace))
        self.velocity_old.assign(fdrk.interpolate(velocity_t0, self.CG_vectorspace))
        
        # Set initial time step 
        self.time_step = compute_time_step(self.problem.domain, \
                                        self.displacement_old, \
                                        self.problem.parameters, 
                                        coeff_cfl=coeff_cfl)
        
        self.time_old = fdrk.Constant(0)
        self.time_midpoint = fdrk.Constant(self.time_step/2)
        self.time_new = fdrk.Constant(self.time_step)
        self.actual_time = fdrk.Constant(0)

        # Boundary conditions 
        dict_essential = problem.get_essential_bcs(self.time_new)

        disp_bc_data = dict_essential["displacement"]
        bcs_displacement = []
        for item in disp_bc_data.items():
            id_disp_bc = item[0]
            value_disp_bc = item[1]

            bc_disp_item = fdrk.DirichletBC(self.CG_vectorspace, value_disp_bc, id_disp_bc)
            bcs_displacement.append(bc_disp_item)

        vel_bc_data = dict_essential["velocity"]
        bcs_velocity = []
        
        for item in vel_bc_data.items():
            id_vel_bc = item[0]
            value_vel_bc = item[1]

            bc_vel_item = fdrk.DirichletBC(self.CG_vectorspace, value_vel_bc, id_vel_bc)
            bcs_velocity.append(bc_vel_item)


        # Set first value for the deformation gradient via Explicit Euler
        self.displacement_oldmid.assign(self.displacement_old + self.time_step/2*self.velocity_old)

        self.time_displacement_old = fdrk.Constant(self.time_step/2)
        self.time_displacement_midpoint = fdrk.Constant(self.time_step)
        self.time_displacement_new = fdrk.Constant(self.time_step + self.time_step/2)
        self.actual_time_displacement = fdrk.Constant(self.time_step/2)

        # Set non linear solver for the displacement

        traction_data_midpoint = problem.get_natural_bcs(self.time_midpoint)

        a_velocity = fdrk.inner(test_CG, rho*trial_CG)*fdrk.dx
        l_velocity = fdrk.inner(test_CG, rho*self.velocity_old)*fdrk.dx \
            - self.time_step * fdrk.inner(fdrk.grad(test_CG), self.first_piola_stress_oldmid)*fdrk.dx \
            + self.time_step * natural_control_follower(test_CG, self.displacement_new, traction_data_midpoint)
                  
        linear_problem_velocity = fdrk.LinearVariationalProblem(a_velocity, 
                                                            l_velocity,
                                                            self.velocity_new, 
                                                            bcs=bcs_velocity)
        
        self.velocity_solver = fdrk.LinearVariationalSolver(linear_problem_velocity,
                                                solver_parameters=solver_parameters)
    
        self.energy_old = 0.5 * fdrk.inner(self.velocity_old, rho*self.velocity_old)*fdrk.dx \
                        + 0.5 * fdrk.inner(self.green_lagrange_strain_old, self.second_piola_stress_old)*fdrk.dx

        self.energy_new = 0.5 * fdrk.inner(self.velocity_new, rho*self.velocity_new)*fdrk.dx \
                        + 0.5 * fdrk.inner(self.green_lagrange_strain_new, self.second_piola_stress_new)*fdrk.dx
        

    def advance(self):
        
        self.velocity_solver.solve()
        self.displacement_newmid.assign(self.displacement_oldmid 
                                    + self.time_step*self.velocity_new)
        self.actual_time.assign(self.time_new)
        

    def update_variables(self):
        self.displacement_old.assign(self.displacement_new)
        self.displacement_oldmid.assign(self.displacement_newmid)
        self.velocity_old.assign(self.velocity_new)

        self.time_step.assign(compute_time_step(self.problem.domain, \
                                        self.displacement_old, \
                                        self.problem.parameters,
                                        coeff_cfl=self.coeff_clf))

        self.time_old.assign(self.actual_time)
        self.time_midpoint.assign(float(self.time_old) + self.time_step/2)
        self.time_new.assign(float(self.time_old) + self.time_step)

        self.time_displacement_old.assign(self.actual_time_displacement)
        self.time_displacement_new.assign(float(self.time_displacement_old) + self.time_step)



    def compute_displaced_mesh(self):
        displaced_coordinates = fdrk.interpolate(self.problem.coordinates_mesh 
                            + self.displacement_old, self.CG_vectorspace)

        return fdrk.Mesh(displaced_coordinates)
    

    def __str__(self):
        return "NonlinearLagrangianStormerVerletSolver"
    
    




    