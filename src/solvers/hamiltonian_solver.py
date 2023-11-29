import firedrake as fdrk
from src.problems.problem import Problem
from src.operators.elastodynamics import HamiltonianElastodynamics
from firedrake.petsc import PETSc

class HamiltonianLinearSolver:
    def __init__(self,
                 problem: Problem,
                 model, 
                 time_step: float,
                 pol_degree= 1,
                 solver_parameters_def_grad={},
                 solver_parameters_energy={}):
        
        self.problem = problem
        self.pol_degree = pol_degree        
        self.time_step = time_step
        self.solver_parameters_def_grad = solver_parameters_def_grad
        self.solver_parameters_energy = solver_parameters_energy


        assert model=="Elastodynamics" or model=="von Karman"

        match model:
            case "Elastodynamics":
                self.operators = HamiltonianElastodynamics(problem, pol_degree)

            case "von Karman":
                raise NotImplementedError("von Karman model yet not implemented.")

        self._set_spaces()
        self._set_initial_boundary_conditions()
        self._energy()
        self._power_balance()
        self._set_solver()
        self._first_step_def_gradient()


    def _set_spaces(self):
        space_energy = self.operators.space_energy
        space_def_gradient = self.operators.space_def_gradient

        self.tests_energy = fdrk.TestFunctions(space_energy)
        self.trials_energy = fdrk.TrialFunctions(space_energy)

        self.state_energy_old = fdrk.Function(space_energy)
        self.state_energy_new = fdrk.Function(space_energy)
        self.state_energy_midpoint = fdrk.Function(space_energy)

        self.test_def_gradient = fdrk.TestFunction(space_def_gradient)
        self.trial_def_gradient = fdrk.TrialFunction(space_def_gradient)

        self.state_def_gradient_old = fdrk.Function(space_def_gradient)
        self.state_def_gradient_new = fdrk.Function(space_def_gradient)
        self.state_def_gradient_midpoint = fdrk.Function(space_def_gradient)

        if isinstance(self.operators, HamiltonianElastodynamics):
            space_displacement = self.operators.space_energy.sub(0)
            self.displacement_old = fdrk.Function(space_displacement)
            self.displacement_new = fdrk.Function(space_displacement)



    def _set_initial_boundary_conditions(self):
        expr_t0 = self.problem.get_initial_conditions()

        interpolated_state_t0 = self.operators.interpolated_initial_conditions(expr_t0)

        if isinstance(self.operators, HamiltonianElastodynamics):
            self.state_def_gradient_old.assign(interpolated_state_t0["def_gradient"])
            self.state_energy_old.sub(0).assign(interpolated_state_t0["velocity"])
            self.state_energy_old.sub(1).assign(interpolated_state_t0["stress"])
            self.displacement_old.assign(interpolated_state_t0["displacement"])

        self.state_energy_midpoint.assign(self.state_energy_old)
        self.state_energy_new.assign(self.state_energy_old)

        self.time_energy_old = fdrk.Constant(0)
        self.time_energy_midpoint = fdrk.Constant(self.time_step/2)
        self.time_energy_new = fdrk.Constant(self.time_step)
        self.actual_time_energy = fdrk.Constant(0)

        self.essential_bcs, self.natural_bcs = self.operators.boundary_conditions(self.time_energy_new, \
                                                                                  self.time_energy_midpoint)



    def _energy(self):
        states_energy_old = self.state_energy_old.subfunctions
        states_energy_new = self.state_energy_new.subfunctions

        self.energy_old = 0.5*self.operators.mass_energy(states_energy_old, states_energy_old)
        self.energy_new = 0.5*self.operators.mass_energy(states_energy_new, states_energy_new)


    def _power_balance(self):

        if isinstance(self.operators, HamiltonianElastodynamics):

            self.power_balance = self.operators.control(self.state_energy_midpoint.subfunctions, 
                                                     self.state_def_gradient_old, 
                                                     self.natural_bcs)
            
        else:
            raise NotImplementedError("Power balance for von Karman problem not yet defined.")
      

    def _set_solver(self):

        operator_def_gradient = self.operators.operator_def_gradient(self.test_def_gradient, \
                                                                       self.trial_def_gradient)
        
        functional_def_gradient = self.operators.functional_def_gradient(self.time_step, \
                                                                         self.test_def_gradient, 
                                                                         self.state_def_gradient_old, \
                                                                         self.state_energy_new.sub(0))
        
        operator_energy = self.operators.operator_energy(self.time_step, \
                                                         self.tests_energy, \
                                                        self.trials_energy, \
                                                        self.state_def_gradient_old)

        functional_energy = self.operators.functional_energy(self.time_step, \
                                                             self.tests_energy, \
                                                            self.state_energy_old.subfunctions, \
                                                            self.state_def_gradient_old, \
                                                            self.natural_bcs)

        linear_def_gradient_problem = fdrk.LinearVariationalProblem(operator_def_gradient, \
                                                                    functional_def_gradient, \
                                                                    self.state_def_gradient_new)
        
        self.linear_def_gradient_solver = fdrk.LinearVariationalSolver(linear_def_gradient_problem, \
                                                                       solver_parameters=self.solver_parameters_def_grad)
        
        linear_energy_problem = fdrk.LinearVariationalProblem(operator_energy, \
                                                                functional_energy, \
                                                                self.state_energy_new, \
                                                                self.essential_bcs)
        
        self.linear_energy_solver = fdrk.LinearVariationalSolver(linear_energy_problem, \
                                                                solver_parameters=self.solver_parameters_energy)
        
        
    def _first_step_def_gradient(self):
        self.linear_def_gradient_solver.solve()

        self.state_def_gradient_midpoint.assign(0.5*(self.state_def_gradient_new + self.state_def_gradient_old))

        self.state_def_gradient_old.assign(self.state_def_gradient_midpoint)

        self.time_def_gradient_old= fdrk.Constant(self.time_step/2)
        self.time_def_gradient_midpoint = fdrk.Constant(self.time_step)
        self.time_def_gradient_new = fdrk.Constant(self.time_step + self.time_step/2)
        self.actual_time_def_gradient = fdrk.Constant(self.time_step/2)


    def integrate(self):

        # First the energy system is advanced 
        self.linear_energy_solver.solve()
        self.linear_def_gradient_solver.solve()

        self.state_energy_midpoint.assign(0.5*(self.state_energy_old + self.state_energy_new))
        self.state_def_gradient_midpoint.assign(0.5*(self.state_def_gradient_old + self.state_def_gradient_new))

        self.actual_time_energy.assign(self.time_energy_new)
        self.actual_time_def_gradient.assign(self.time_def_gradient_new)

        if isinstance(self.operators, HamiltonianElastodynamics):

            self.displacement_new.assign(self.displacement_old + self.time_step * self.state_energy_midpoint.sub(0))



    def update_variables(self):
        self.state_energy_old.assign(self.state_energy_new)
        self.state_def_gradient_old.assign(self.state_def_gradient_new)
        self.displacement_old.assign(self.displacement_new)

        self.time_energy_old.assign(self.actual_time_energy)
        self.time_energy_midpoint.assign(float(self.time_energy_old) + self.time_step/2)
        self.time_energy_new.assign(float(self.time_energy_old) + self.time_step)

        self.time_def_gradient_old.assign(self.actual_time_def_gradient)
        self.time_def_gradient_midpoint.assign(float(self.time_def_gradient_old) + self.time_step/2)
        self.time_def_gradient_new.assign(float(self.time_def_gradient_old) + self.time_step)


    def __str__(self):
        return "HamiltonianLinearSolver"