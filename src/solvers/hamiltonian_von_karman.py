import firedrake as fdrk
from src.problems.problem import Problem
from src.tools.von_karman import mass_form_energy, operator_energy, functional_energy


class HamiltonianVonKarmanSolver:
    def __init__(self,
                 problem: Problem,
                 time_step: float,
                 pol_degree= 1,
                 solver_parameters_energy={}):
        
        self.coordinates_mesh = problem.coordinates_mesh
        self.time_step = time_step
        self.problem = problem

        deg_eps = 2*(pol_degree - 1)

        space_mem_velocity = fdrk.VectorFunctionSpace(problem.domain, "CG", deg_eps+1)
        space_bend_velocity = fdrk.FunctionSpace(problem.domain, "CG", pol_degree)

        # space_mem_stress = fdrk.TensorFunctionSpace(problem.domain, "DG", deg_eps, symmetry=True)
        space_mem_stress = fdrk.FunctionSpace(problem.domain, "Regge", deg_eps)
        space_bend_stress = fdrk.FunctionSpace(problem.domain, "HHJ", pol_degree - 1)

        self.space_bend_displacement = fdrk.FunctionSpace(problem.domain, "CG", pol_degree)

        space_energy = space_mem_velocity * space_bend_velocity * space_mem_stress * space_bend_stress

        tests_energy = fdrk.TestFunctions(space_energy)
        trials_energy = fdrk.TrialFunctions(space_energy)

        self.state_energy_old = fdrk.Function(space_energy)
        self.state_energy_new = fdrk.Function(space_energy)
        self.state_energy_midpoint = fdrk.Function(space_energy)

        self.test_bend_displacement = fdrk.TestFunction(self.space_bend_displacement)
        self.trial_bend_displacement = fdrk.TrialFunction(self.space_bend_displacement)

        self.bend_displacement_old = fdrk.Function(self.space_bend_displacement)
        self.bend_displacement_new = fdrk.Function(self.space_bend_displacement)
        self.bend_displacement_midpoint = fdrk.Function(self.space_bend_displacement)

        expr_t0 = problem.get_initial_conditions()

        bend_displacement_exp = expr_t0["bending displacement"]

        mem_velocity_exp = expr_t0["membrane velocity"]
        bend_velocity_exp = expr_t0["bending velocity"]

        mem_stress_exp = expr_t0["membrane stress"]
        bend_stress_exp = expr_t0["bending stress"]

        bend_displacement_t0 = fdrk.interpolate(bend_displacement_exp, self.space_bend_displacement)

        mem_velocity_t0 = fdrk.interpolate(mem_velocity_exp, space_energy.sub(0))
        bend_velocity_t0 = fdrk.interpolate(bend_velocity_exp, space_energy.sub(1))

        mem_stress_t0 = fdrk.interpolate(mem_stress_exp, space_energy.sub(2))
        bend_stress_t0 = fdrk.interpolate(bend_stress_exp, space_energy.sub(3))
        
        self.bend_displacement_old.assign(bend_displacement_t0)

        self.state_energy_old.sub(0).assign(mem_velocity_t0)
        self.state_energy_old.sub(1).assign(bend_velocity_t0)

        self.state_energy_old.sub(2).assign(mem_stress_t0)
        self.state_energy_old.sub(3).assign(bend_stress_t0)

        self.state_energy_midpoint.assign(self.state_energy_old)
        self.state_energy_new.assign(self.state_energy_old)

        self.time_energy_old = fdrk.Constant(0)
        self.time_energy_midpoint = fdrk.Constant(self.time_step/2)
        self.time_energy_new = fdrk.Constant(self.time_step)
        self.actual_time_energy = fdrk.Constant(0)

        dict_essential = problem.get_essential_bcs(self.time_energy_new)

        # bend_displacement_bc_data = dict_essential["bending displacement"]
        # bend_displacement_bcs = [fdrk.DirichletBC(self.space_bend_displacement, item[1], item[0]) \
        #                         for item in bend_displacement_bc_data.items()]
        
        mem_velocity_bc_data = dict_essential["membrane velocity"]
        mem_velocity_bcs = [fdrk.DirichletBC(space_energy.sub(0), item[1], item[0]) \
                                for item in mem_velocity_bc_data.items()]
        
        bend_velocity_bc_data = dict_essential["bending velocity"]
        bend_velocity_bcs = [fdrk.DirichletBC(space_energy.sub(1), item[1], item[0]) \
                                for item in bend_velocity_bc_data.items()]
        
        bend_stress_bc_data = dict_essential["bending stress"]
        bend_stress_bcs = [fdrk.DirichletBC(space_energy.sub(3), item[1], item[0]) \
                                for item in bend_stress_bc_data.items()]

        # Set first value for the deformation gradient via Explicit Euler
        self.bend_displacement_midpoint.assign(self.bend_displacement_old \
                                          + self.time_step/2*self.state_energy_old.sub(1))
        
        self.bend_displacement_old.assign(self.bend_displacement_midpoint)

        self.time_displacement_old= fdrk.Constant(self.time_step/2)
        self.time_displacement_midpoint = fdrk.Constant(self.time_step)
        self.time_displacement_new = fdrk.Constant(self.time_step + self.time_step/2)
        self.actual_time_displacement = fdrk.Constant(self.time_step/2)

        # Set solver for the energy part
        states_energy_old = self.state_energy_old.subfunctions
        states_energy_new = self.state_energy_new.subfunctions
        self.energy_old = 0.5*mass_form_energy(states_energy_old, states_energy_old, problem.parameters)
        self.energy_new = 0.5*mass_form_energy(states_energy_new, states_energy_new, problem.parameters)

        a_form = operator_energy(self.time_step, \
                                tests_energy, \
                                trials_energy, \
                                self.bend_displacement_old, \
                                problem.parameters, \
                                problem.normal_versor)

        l_form = functional_energy(self.time_step, \
                                    tests_energy, \
                                    self.state_energy_old.subfunctions, \
                                    self.bend_displacement_old, \
                                    problem.parameters, \
                                    problem.normal_versor)

        linear_energy_problem = fdrk.LinearVariationalProblem(a_form, \
                                                            l_form, \
                                                            self.state_energy_new, \
                                                            mem_velocity_bcs +\
                                                            bend_velocity_bcs +\
                                                            bend_stress_bcs)
    
        self.linear_energy_solver = fdrk.LinearVariationalSolver(linear_energy_problem, \
                                                                solver_parameters=solver_parameters_energy)


    def integrate(self):

        # First the energy system is advanced at n+1
        self.linear_energy_solver.solve()
        self.state_energy_midpoint.assign(0.5*(self.state_energy_old + self.state_energy_new))

        # Compute solution for displacement at n+3∕2
        self.bend_displacement_new.assign(self.bend_displacement_old + self.time_step * self.state_energy_new.sub(1))

        self.bend_displacement_midpoint.assign(0.5*(self.bend_displacement_old + \
                                                    self.bend_displacement_new))

        self.actual_time_energy.assign(self.time_energy_new)
        self.actual_time_displacement.assign(self.time_displacement_new)


    def update_variables(self):
        self.state_energy_old.assign(self.state_energy_new)
        self.bend_displacement_old.assign(self.bend_displacement_new)

        self.time_energy_old.assign(self.actual_time_energy)
        self.time_energy_midpoint.assign(float(self.time_energy_old) + self.time_step/2)
        self.time_energy_new.assign(float(self.time_energy_old) + self.time_step)

        self.time_displacement_old.assign(self.actual_time_displacement)
        self.time_displacement_midpoint.assign(float(self.time_displacement_old) + self.time_step/2)
        self.time_displacement_new.assign(float(self.time_displacement_old) + self.time_step)


    def __str__(self):
        return "HamiltonianVonKarmanSolver"