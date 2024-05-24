import firedrake as fdrk
from src.problems.problem import DynamicProblem
from src.tools.elasticity import natural_control_follower, \
                                operator_energy, functional_energy,\
                                stiffness_tensor, mass_form_energy
from src.tools.common import compute_time_step

class HamiltonianSaintVenantSolverStaticCondensation:
    def __init__(self,
                problem: DynamicProblem,
                pol_degree= 1,
                coeff_cfl = 0.8,
                solver_parameters_energy={}):
        
        self.coordinates_mesh = problem.coordinates_mesh
        self.problem = problem
        self.coeff_cfl = coeff_cfl

        self.density = self.problem.parameters["rho"]
        self.E = self.problem.parameters["E"]
        self.nu = self.problem.parameters["nu"]
        
        displ_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)
        space_stress_tensor = fdrk.TensorFunctionSpace(problem.domain, "DG", pol_degree-1, \
                                                    symmetry=True)

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

        expr_t0 = problem.get_initial_conditions()

        displacement_exp = expr_t0["displacement"]
        velocity_exp = expr_t0["velocity"]
        strain_exp = expr_t0["strain"]

        stress_exp = stiffness_tensor(strain_exp, self.E, self.nu)

        displacement_t0 = fdrk.interpolate(displacement_exp, self.space_displacement)
        velocity_t0 = fdrk.interpolate(velocity_exp, space_energy.sub(0))
        stress_t0 = fdrk.interpolate(stress_exp, space_energy.sub(1))
        
        self.displacement_old.assign(displacement_t0)
        self.state_energy_old.sub(0).assign(velocity_t0)
        self.state_energy_old.sub(1).assign(stress_t0)

        self.state_energy_midpoint.assign(self.state_energy_old)
        self.state_energy_new.assign(self.state_energy_old)
        
        # Compute time step using wave speed for non linear behaviour
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

        velocity_bc_data = dict_essential["velocity"]
        velocity_bcs = [fdrk.DirichletBC(displ_vectorspace, item[1], item[0]) \
                        for item in velocity_bc_data.items()]

        natural_bcs = problem.get_natural_bcs(self.time_midpoint)

        # Set first value for the deformation gradient via Explicit Euler
        self.displacement_old.assign(self.displacement_old + self.time_step/2*self.state_energy_old.sub(0))

        self.time_displacement_old= fdrk.Constant(self.time_step/2)
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
        
        # Apply static condensation to solve for velocity only

        A_tensor = fdrk.Tensor(a_form) 
        self.A_blocks = A_tensor.blocks

        A_velocity = self.A_blocks[0, 0] \
            - self.A_blocks[0, 1] * self.A_blocks[1, 1].inv * self.A_blocks[1, 0]

        b_vector = fdrk.Tensor(l_form)
        self.b_blocks = b_vector.blocks

        b_velocity = self.b_blocks[0] \
            - self.A_blocks[0, 1] * self.A_blocks[1, 1].inv * self.b_blocks[1]

        self.velocity_new = fdrk.Function(displ_vectorspace)
        linear_velocity_problem  = fdrk.LinearVariationalProblem(A_velocity, b_velocity, \
                                                                self.velocity_new, \
                                                                velocity_bcs)

        self.linear_velocity_solver = fdrk.LinearVariationalSolver(linear_velocity_problem, \
                                    solver_parameters=solver_parameters_energy)


    def advance(self):

        # First the energy system is advanced at n+1
        self.linear_velocity_solver.solve()

        self._assemble_solution()

        self.state_energy_midpoint.assign(0.5*(self.state_energy_old + self.state_energy_new))

        # Compute solution for displacement at n+3∕2
        self.displacement_new.assign(self.displacement_old + self.time_step * self.state_energy_new.sub(0))

        self.actual_time.assign(self.time_new)
        self.actual_time_displacement.assign(self.time_displacement_new)


    def update_variables(self):
        self.state_energy_old.assign(self.state_energy_new)
        self.displacement_old.assign(self.displacement_new)

        self.time_step.assign(compute_time_step(self.problem.domain, \
                                        self.displacement_old, \
                                        self.problem.parameters,
                                        coeff_cfl=self.coeff_cfl))

        self.time_old.assign(self.actual_time)
        self.time_midpoint.assign(float(self.time_old) + self.time_step/2)
        self.time_new.assign(float(self.time_old) + self.time_step)

        self.time_displacement_old.assign(self.actual_time_displacement)
        self.time_displacement_new.assign(float(self.time_displacement_old) + self.time_step)


    
    def _assemble_solution(self):

        local_velocity = fdrk.AssembledVector(self.velocity_new)

        stress_new = fdrk.assemble(self.A_blocks[1, 1].inv * (self.b_blocks[1] \
                                    - self.A_blocks[1, 0] * local_velocity))
        
        coeff_stress = stress_new.vector().get_local()

        self.state_energy_new.sub(0).assign(self.velocity_new)
        self.state_energy_new.sub(1).vector().set_local(coeff_stress)


    def compute_displaced_mesh(self):
        displaced_coordinates = fdrk.interpolate(self.coordinates_mesh 
                                            + self.displacement_old, self.space_displacement)

        return fdrk.Mesh(displaced_coordinates)


    def __str__(self):
        return "HamiltonianSaintVenantSolverStaticCondensation"