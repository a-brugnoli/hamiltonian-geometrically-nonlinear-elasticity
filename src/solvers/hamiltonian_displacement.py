import firedrake as fdrk
from src.problems.problem import Problem
from firedrake.petsc import PETSc
from utils.utils_elasticity import mass_form_energy, natural_control_follower, \
                                operator_energy, functional_energy

from utils.common_utils import compute_min_mesh_size

import numpy as np

class HamiltonianDisplacementSolver:
    def __init__(self,
                 problem: Problem,
                 time_step: float,
                 pol_degree= 1,
                 solver_parameters_energy={}):
        
        self.coordinates_mesh = problem.coordinates_mesh
        self.time_step = time_step
        self.problem = problem
        self.jacobian_inverse = fdrk.JacobianInverse(self.problem.domain)
        self.delta_x_min = compute_min_mesh_size(problem.domain)
        self.CG1_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "CG", 1)
        self.cfl_vectorfield = fdrk.Function(self.CG1_vectorspace)

        # if problem.domain.ufl_cell().is_simplex():
        #     displ_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)
        # else:
        #     displ_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "S", pol_degree)

        displ_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)
        # space_stress_tensor = fdrk.TensorFunctionSpace(problem.domain, "DG", pol_degree-1, symmetry=True)

        cell = problem.domain.ufl_cell()

        regge_broken_fe = fdrk.BrokenElement(fdrk.FiniteElement("Regge", cell, pol_degree-1))
        space_stress_tensor = fdrk.FunctionSpace(problem.domain, regge_broken_fe)

        # space_stress_tensor = fdrk.FunctionSpace(problem.domain, "Regge", pol_degree-1)

        self.space_displacement = displ_vectorspace
        space_energy = displ_vectorspace * space_stress_tensor

        tests_energy = fdrk.TestFunctions(space_energy)
        trials_energy = fdrk.TrialFunctions(space_energy)

        self.state_energy_old = fdrk.Function(space_energy)
        self.state_energy_new = fdrk.Function(space_energy)
        self.state_energy_midpoint = fdrk.Function(space_energy)

        self.test_displacement = fdrk.TestFunction(self.space_displacement)
        self.trial_displacement = fdrk.TrialFunction(self.space_displacement)

        self.displacement_old = fdrk.Function(self.space_displacement)
        self.displacement_new = fdrk.Function(self.space_displacement)
        self.displacement_midpoint = fdrk.Function(self.space_displacement)

        space_DG = fdrk.TensorFunctionSpace(problem.domain, "DG", pol_degree-1)
        self.deformation_gradient_new = fdrk.Function(space_DG)

        expr_t0 = problem.get_initial_conditions()

        velocity_exp = expr_t0["velocity"]
        stress_exp = expr_t0["stress"]
        displacement_exp = expr_t0["displacement"]

        displacement_t0 = fdrk.interpolate(displacement_exp, self.space_displacement)
        velocity_t0 = fdrk.interpolate(velocity_exp, space_energy.sub(0))
        stress_t0 = fdrk.interpolate(stress_exp, space_energy.sub(1))
        
        self.displacement_old.assign(displacement_t0)
        self.state_energy_old.sub(0).assign(velocity_t0)
        self.state_energy_old.sub(1).assign(stress_t0)

        self.state_energy_midpoint.assign(self.state_energy_old)
        self.state_energy_new.assign(self.state_energy_old)

        self.time_energy_old = fdrk.Constant(0)
        self.time_energy_midpoint = fdrk.Constant(self.time_step/2)
        self.time_energy_new = fdrk.Constant(self.time_step)
        self.actual_time_energy = fdrk.Constant(0)

        dict_essential = problem.get_essential_bcs(self.time_energy_new)

        velocity_bc_data = dict_essential["velocity"]
        velocity_bcs = [fdrk.DirichletBC(space_energy.sub(0), item[1], item[0]) \
                        for item in velocity_bc_data.items()]

        # displacement_bc_data = dict_essential["displacement"]
        # displacement_bcs = [fdrk.DirichletBC(space_displacement, item[1], item[0]) for item in displacement_bc_data.items()]

        natural_bcs = problem.get_natural_bcs(self.time_energy_midpoint)

        # Set first value for the deformation gradient via Explicit Euler
        self.displacement_midpoint.assign(self.displacement_old + self.time_step/2*self.state_energy_old.sub(0))
        self.displacement_old.assign(self.displacement_midpoint)

        self.time_displacement_old= fdrk.Constant(self.time_step/2)
        self.time_displacement_midpoint = fdrk.Constant(self.time_step)
        self.time_displacement_new = fdrk.Constant(self.time_step + self.time_step/2)
        self.actual_time_displacement = fdrk.Constant(self.time_step/2)

        # Set solver for the energy part
        states_energy_old = self.state_energy_old.subfunctions
        states_energy_new = self.state_energy_new.subfunctions
        self.energy_old = 0.5*mass_form_energy(states_energy_old, states_energy_old, problem.parameters)
        self.energy_new = 0.5*mass_form_energy(states_energy_new, states_energy_new, problem.parameters)

        self.power_balance = natural_control_follower(self.state_energy_midpoint.subfunctions[0], 
                                                     self.displacement_old, 
                                                     natural_bcs)

        
        a_form = operator_energy(self.time_step, \
                                tests_energy, \
                                trials_energy, \
                                self.displacement_old, \
                                problem.parameters)

        l_form = functional_energy(self.time_step, \
                                    tests_energy, \
                                    self.state_energy_old.subfunctions, \
                                    self.displacement_old, \
                                    natural_bcs, 
                                    problem.parameters)

        linear_energy_problem = fdrk.LinearVariationalProblem(a_form, \
                                                                l_form, \
                                                                self.state_energy_new, \
                                                                velocity_bcs)
        
        self.linear_energy_solver = fdrk.LinearVariationalSolver(linear_energy_problem, \
                                                                solver_parameters=solver_parameters_energy)


    def integrate(self):

        # First the energy system is advanced at n+1
        self.linear_energy_solver.solve()
        self.state_energy_midpoint.assign(0.5*(self.state_energy_old + self.state_energy_new))

        # Compute solution for displacement at n+3∕2
        self.displacement_new.assign(self.displacement_old + self.time_step * self.state_energy_new.sub(0))

        self.displacement_midpoint.assign(0.5*(self.displacement_old + self.displacement_new))

        self.actual_time_energy.assign(self.time_energy_new)
        self.actual_time_displacement.assign(self.time_displacement_new)


    def get_wave_cfl(self):
        rho = float(self.problem.parameters["Density"])
        E = float(self.problem.parameters["Young modulus"])
        nu = float(self.problem.parameters["Poisson ratio"])

        return self.time_step/self.delta_x_min*np.sqrt(E/rho*((1-nu**2)))


    def get_dinamic_cfl(self):

        velocity_old = self.state_energy_old.subfunctions[00]

        self.cfl_vectorfield.assign(fdrk.interpolate(self.time_step * fdrk.dot(self.jacobian_inverse,\
                            velocity_old), self.CG1_vectorspace))

        coeff_cfl = np.amax(np.abs(self.cfl_vectorfield.dat.data), axis=1)

        return np.max(coeff_cfl)

    def update_variables(self):
        self.state_energy_old.assign(self.state_energy_new)
        self.displacement_old.assign(self.displacement_new)

        self.time_energy_old.assign(self.actual_time_energy)
        self.time_energy_midpoint.assign(float(self.time_energy_old) + self.time_step/2)
        self.time_energy_new.assign(float(self.time_energy_old) + self.time_step)

        self.time_displacement_old.assign(self.actual_time_displacement)
        self.time_displacement_midpoint.assign(float(self.time_displacement_old) + self.time_step/2)
        self.time_displacement_new.assign(float(self.time_displacement_old) + self.time_step)


    def output_displaced_mesh(self):
        displaced_coordinates = fdrk.interpolate(self.coordinates_mesh 
                                            + self.displacement_old, self.space_displacement)

        return fdrk.Mesh(displaced_coordinates)


    def __str__(self):
        return "HamiltonianDisplacementSolver"