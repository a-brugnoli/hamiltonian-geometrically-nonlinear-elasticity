import firedrake as fdrk
from src.problems.problem import Problem
from firedrake.petsc import PETSc

import numpy as np


class HamiltonianNeoHookeanSolver:
    def __init__(self,
                 problem: Problem,
                 time_step: float,
                 pol_degree= 1,
                 solver_parameters={}):
        

        self.coordinates_mesh = problem.coordinates_mesh
        self.time_step = time_step
        self.problem = problem
        self.dim = problem.dim
        
        self.space_displacement = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)
        self.space_strain = fdrk.FunctionSpace(problem.domain, "Regge", pol_degree - 1)
        self.space_stress = self.space_strain

        space_energy = self.space_displacement * self.space_strain * self.space_stress
        self.test_velocity, self.test_strain, self.test_stress = fdrk.TestFunctions(space_energy)

        self.state_energy_old = fdrk.Function(space_energy)
        self.state_energy_new = fdrk.Function(space_energy)

        self.test_displacement = fdrk.TestFunction(self.space_displacement)
        self.trial_displacement = fdrk.TrialFunction(self.space_displacement)

        self.displacement_old = fdrk.Function(self.space_displacement)
        self.displacement_new = fdrk.Function(self.space_displacement)

        expr_t0 = problem.get_initial_conditions()

        displacement_exp = expr_t0["displacement"]
        velocity_exp = expr_t0["velocity"]
        strain_exp = expr_t0["strain"]

        displacement_t0 = fdrk.interpolate(displacement_exp, self.space_displacement)
        velocity_t0 = fdrk.interpolate(velocity_exp, space_energy.sub(0))
        strain_t0 = fdrk.interpolate(strain_exp, space_energy.sub(1))
        
        self.displacement_old.assign(displacement_t0)
        self.state_energy_old.sub(0).assign(velocity_t0)
        self.state_energy_old.sub(1).assign(strain_t0)

        self.state_energy_new.assign(self.state_energy_old)

        # Set first value for the deformation gradient via Explicit Euler
        self.displacement_old.assign(self.displacement_old + self.time_step/2*self.state_energy_old.sub(0))

        self.time_displacement_old= fdrk.Constant(self.time_step/2)
        self.time_displacement_new = fdrk.Constant(self.time_step + self.time_step/2)
        self.actual_time_displacement = fdrk.Constant(self.time_step/2)

        self.time_energy_old = fdrk.Constant(0)
        self.time_energy_new = fdrk.Constant(self.time_step)
        self.actual_time_energy = fdrk.Constant(0)

        # Boundary conditions

        dict_essential = problem.get_essential_bcs(self.time_energy_new)

        velocity_bc_data = dict_essential["velocity"]
        velocity_bcs = [fdrk.DirichletBC(space_energy.sub(0), item[1], item[0]) \
                        for item in velocity_bc_data.items()]
        
        # Set solver for the energy part
        self.velocity_old, self.strain_old, self.stress_old = self.state_energy_old.subfunctions
        self.velocity_new, self.strain_new, self.stress_new = fdrk.split(self.state_energy_new)

        self.velocity_midpoint = 0.5*(self.velocity_new + self.velocity_old)
        self.strain_midpoint = 0.5*(self.strain_new + self.strain_old)
        
        rho = self.problem.parameters["rho"]
        E = self.problem.parameters["E"]
        nu = self.problem.parameters["nu"]

        F_midpoint = fdrk.Identity(self.dim) + fdrk.grad(self.displacement_old)
        residual_vel_eq = fdrk.inner(self.test_velocity, rho*(self.velocity_new - self.velocity_old)/self.time_step)*fdrk.dx \
        + fdrk.inner(fdrk.grad(self.test_velocity), fdrk.dot(F_midpoint, self.strain_midpoint))*fdrk.dx
        
        residual_strain_eq = fdrk.inner(self.test_strain, (self.strain_new - self.strain_old)/self.time_step) * fdrk.dx \
        - fdrk.inner(self.test_strain, fdrk.dot(F_midpoint.T, fdrk.grad(self.velocity_midpoint))) * fdrk.dx

        caunchy_new = 2*self.strain_new + fdrk.Identity(self.dim) 
        psi = energy_density_neo_hookean(caunchy_new, E, nu)
        deformation_energy = psi * fdrk.dx
        residual_stress_eq = fdrk.inner(self.test_strain, self.stress_new) * fdrk.dx \
            - fdrk.derivative(deformation_energy, self.strain_new, self.test_strain)
        
        residual = residual_vel_eq + residual_strain_eq + residual_stress_eq

        nonlinear_problem = fdrk.NonlinearVariationalSolver(residual, self.state_energy_new, velocity_bcs)

        self.nonlinear_solver = fdrk.NonlinearVariationalSolver(nonlinear_problem, solver_parameters)



    def integrate(self):
        # First the energy system is advanced at n+1
        self.nonlinear_solver.solve()
        # Compute solution for displacement at n+3∕2
        self.displacement_new.assign(self.displacement_old + self.time_step * self.state_energy_new.sub(0))


        self.actual_time_energy.assign(self.time_energy_new)
        self.actual_time_displacement.assign(self.time_displacement_new)


    def update_variables(self):
        self.state_energy_old.assign(self.state_energy_new)
        self.displacement_old.assign(self.displacement_new)

        self.time_energy_old.assign(self.actual_time_energy)
        self.time_energy_new.assign(float(self.time_energy_old) + self.time_step)

        self.time_displacement_old.assign(self.actual_time_displacement)
        self.time_displacement_new.assign(float(self.time_displacement_old) + self.time_step)


    def output_displaced_mesh(self):
        displaced_coordinates = fdrk.interpolate(self.coordinates_mesh 
                                            + self.displacement_old, self.space_displacement)

        return fdrk.Mesh(displaced_coordinates)


    def energy(self, velocity, strain):

        rho = self.problem.parameters["rho"]
        E = self.problem.parameters["E"]
        nu = self.problem.parameters["nu"]

        cauchy_strain = 2 * strain + fdrk.Identity(self.dim)

        kinetic_energy = 1/2 * fdrk.inner(velocity, velocity)*fdrk.dx 
        deformation_energy = 1/2 * energy_density_neo_hookean(cauchy_strain, E, nu)*fdrk.dx

        return kinetic_energy + deformation_energy


def energy_density_neo_hookean(cauchy_strain, young_modulus, poisson_ratio):
        
    mu = young_modulus / (2*(1 + poisson_ratio))
    lamda = young_modulus*poisson_ratio/((1 - 2*poisson_ratio)*(1 + poisson_ratio))
    kappa = lamda + 2/3*mu

    J = fdrk.sqrt(fdrk.det(cauchy_strain))

    deviatoric_part = 1/2 * mu * (J**(-2/3) * fdrk.tr(cauchy_strain) - 3)
    volumetric_part = 1/4 * (J**2 - 1) - 1/2*kappa*fdrk.ln(J)

    return deviatoric_part + volumetric_part


