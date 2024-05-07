import firedrake as fdrk
from src.problems.problem import DynamicProblem
from firedrake.petsc import PETSc
from src.tools.common import compute_time_step


class HamiltonianNeoHookeanSolver:
    def __init__(self,
                 problem: DynamicProblem,
                 pol_degree= 1,
                 solver_parameters={}):
        

        self.coordinates_mesh = problem.coordinates_mesh
        self.problem = problem
        self.dim = problem.dim

        self.density = self.problem.parameters["rho"]
        self.mu = self.problem.parameters["mu"]
        self.kappa = self.problem.parameters["kappa"]
        
        self.space_displacement = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)
        self.space_strain = fdrk.FunctionSpace(problem.domain, "Regge", pol_degree - 1)
        # self.space_strain = fdrk.TensorFunctionSpace(problem.domain, "DG", pol_degree - 1, symmetry = True)
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

        cauchy_strain_t0 = 2 * strain_t0 + fdrk.Identity(self.dim)
        stress_t0 = fdrk.interpolate(second_piola_definition(cauchy_strain_t0, \
                                                            self.mu, self.kappa),\
                                                            self.space_stress)

        self.displacement_old.assign(displacement_t0)
        self.state_energy_old.sub(0).assign(velocity_t0)
        self.state_energy_old.sub(1).assign(strain_t0)
        self.state_energy_old.sub(2).assign(stress_t0)

        self.state_energy_new.assign(self.state_energy_old)

        # Set first value for the deformation gradient via Explicit Euler
        self.time_step = compute_time_step(self.problem.domain, \
                                        self.displacement_old, \
                                        self.problem.parameters)
        
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
        self.stress_midpoint = 0.5*(self.stress_new + self.stress_old)

        F_midpoint = fdrk.Identity(self.dim) + fdrk.grad(self.displacement_old)
        residual_vel_eq = fdrk.inner(self.test_velocity, self.density*(self.velocity_new - self.velocity_old)/self.time_step)*fdrk.dx \
        + fdrk.inner(fdrk.grad(self.test_velocity), fdrk.dot(F_midpoint, self.stress_midpoint))*fdrk.dx
        
        residual_strain_eq = fdrk.inner(self.test_strain, (self.strain_new - self.strain_old)/self.time_step) * fdrk.dx \
        - fdrk.inner(self.test_strain, fdrk.dot(F_midpoint.T, fdrk.grad(self.velocity_midpoint))) * fdrk.dx

        cauchy_strain = 2 * self.strain_new + fdrk.Identity(self.dim)
        residual_stress_eq = fdrk.inner(self.test_stress, self.stress_new\
                            - second_piola_definition(cauchy_strain, \
                                                    self.mu, \
                                                    self.kappa)) * fdrk.dx 
                        
        
        residual = residual_vel_eq + residual_strain_eq + residual_stress_eq

        nonlinear_problem = fdrk.NonlinearVariationalProblem(residual, self.state_energy_new, bcs = velocity_bcs)

        self.nonlinear_solver = fdrk.NonlinearVariationalSolver(nonlinear_problem, solver_parameters = solver_parameters)


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

        self.time_step.assign(compute_time_step(self.problem.domain, \
                                        self.displacement_old, \
                                        self.problem.parameters))
        
        self.time_energy_old.assign(self.actual_time_energy)
        self.time_energy_new.assign(float(self.time_energy_old) + self.time_step)

        self.time_displacement_old.assign(self.actual_time_displacement)
        self.time_displacement_new.assign(float(self.time_displacement_old) + self.time_step)


    def output_displaced_mesh(self):
        displaced_coordinates = fdrk.interpolate(self.coordinates_mesh 
                                            + self.displacement_old, self.space_displacement)

        return fdrk.Mesh(displaced_coordinates)


    def energy(self, velocity, strain):
        cauchy_strain = 2 * strain + fdrk.Identity(self.dim)

        kinetic_energy = 1/2 * fdrk.inner(velocity, self.density*velocity)*fdrk.dx 
        deformation_energy = energy_density_neo_hookean(cauchy_strain, \
                                                        self.mu, \
                                                        self.kappa)*fdrk.dx

        return kinetic_energy + deformation_energy

    def __str__(self):
        return "HamiltonianNeoHookeanSolver"


def energy_density_neo_hookean(cauchy_strain, mu, kappa):
        
    J = fdrk.sqrt(fdrk.det(cauchy_strain))

    deviatoric_part = 1/2 * mu * (J**(-2/3) * fdrk.tr(cauchy_strain) - 3)
    volumetric_part = 1/4 * kappa * (J**2 - 1) - 1/2*kappa*fdrk.ln(J)

    return volumetric_part + deviatoric_part


def second_piola_definition(cauchy_strain, mu, kappa):


    J = fdrk.sqrt(fdrk.det(cauchy_strain))

    dpsi_dev_dC = 1/2*mu*J**(-2/3)*(fdrk.Identity(3) \
                                        - 1/3*fdrk.tr(cauchy_strain)*fdrk.inv(cauchy_strain))

    dpsi_vol_dC = 1/4*kappa*(J**2 - 1)*fdrk.inv(cauchy_strain)

    
    return 2*(dpsi_dev_dC + dpsi_vol_dC)
