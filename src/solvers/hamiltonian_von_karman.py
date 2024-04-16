import firedrake as fdrk
from src.problems.problem import Problem
from src.tools.von_karman import mass_form_energy, dynamics_form_energy,\
    operator_energy, functional_energy
from firedrake.petsc import PETSc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

class HamiltonianVonKarmanSolver:
    def __init__(self,
                problem: Problem,
                time_step: float,
                pol_degree= 1,
                solver_parameters_energy={}, 
                membrane_inertia = True,
                coupling = True):
        
        self.coordinates_mesh = problem.coordinates_mesh
        self.time_step = time_step
        self.problem = problem

        space_mem_velocity = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)
        space_mem_stress = fdrk.FunctionSpace(problem.domain, "Regge", pol_degree - 1)
        # space_mem_stress = fdrk.TensorFunctionSpace(problem.domain, "DG", \
                                                    # pol_degree-1, symmetry=True)

        self.space_bend_displacement = fdrk.FunctionSpace(problem.domain, "CG", pol_degree)

        space_bend_velocity = fdrk.FunctionSpace(problem.domain, "CG", pol_degree)
        space_bend_stress = fdrk.FunctionSpace(problem.domain, "HHJ", pol_degree - 1)

        space_energy = space_mem_velocity * space_mem_stress * space_bend_velocity * space_bend_stress

        PETSc.Sys.Print(f"Dimension space membrane : \
                        {space_mem_velocity.dim() + space_mem_stress.dim()}")
        PETSc.Sys.Print(f"Dimension space bending : \
                        {space_bend_velocity.dim() + space_bend_stress.dim()}")

        tests_energy = fdrk.TestFunctions(space_energy)
        trials_energy = fdrk.TrialFunctions(space_energy)

        self.state_energy_old = fdrk.Function(space_energy)
        self.state_energy_new = fdrk.Function(space_energy)

        test_bend_displacement = fdrk.TestFunction(self.space_bend_displacement)
        trial_bend_displacement = fdrk.TrialFunction(self.space_bend_displacement)

        self.bend_displacement_old = fdrk.Function(self.space_bend_displacement)
        self.bend_displacement_new = fdrk.Function(self.space_bend_displacement)

        expr_t0 = problem.get_initial_conditions()

        mem_velocity_exp = expr_t0["membrane velocity"]
        mem_stress_exp = expr_t0["membrane stress"]

        bend_displacement_exp = expr_t0["bending displacement"]
        bend_velocity_exp = expr_t0["bending velocity"]
        bend_stress_exp = expr_t0["bending stress"]


        mem_velocity_t0 = fdrk.interpolate(mem_velocity_exp, space_energy.sub(0))
        mem_stress_t0 = fdrk.interpolate(mem_stress_exp, space_energy.sub(1))

        bend_velocity_t0 = fdrk.interpolate(bend_velocity_exp, space_energy.sub(2))
        bend_stress_t0 = fdrk.interpolate(bend_stress_exp, space_energy.sub(3))

        bend_displacement_t0 = fdrk.interpolate(bend_displacement_exp, self.space_bend_displacement)
        self.bend_displacement_old.assign(bend_displacement_t0)

        self.state_energy_old.sub(0).assign(mem_velocity_t0)
        self.state_energy_old.sub(1).assign(mem_stress_t0)

        self.state_energy_old.sub(2).assign(bend_velocity_t0)
        self.state_energy_old.sub(3).assign(bend_stress_t0)

        self.state_energy_new.assign(self.state_energy_old)

        self.time_energy_old = fdrk.Constant(0)
        self.time_energy_new = fdrk.Constant(self.time_step)
        self.actual_time_energy = fdrk.Constant(0)

        dict_essential = problem.get_essential_bcs(self.time_energy_new)

        try:        
            bend_velocity_bc_data = dict_essential["bending velocity"]
            bend_velocity_bcs = [fdrk.DirichletBC(space_energy.sub(2), item[1], item[0]) \
                                for item in bend_velocity_bc_data.items()]
        except KeyError:
            bend_velocity_bcs = []

        try:
            mem_velocity_bc_data = dict_essential["membrane velocity"]
            mem_velocity_bcs = [fdrk.DirichletBC(space_energy.sub(0), item[1], item[0]) \
                                for item in mem_velocity_bc_data.items()]
        except:
            mem_velocity_bcs = []

        try:
            bend_stress_bc_data = dict_essential["bending stress"]
            bend_stress_bcs = [fdrk.DirichletBC(space_energy.sub(3), item[1], item[0]) \
                                for item in bend_stress_bc_data.items()]
        except KeyError:
            bend_stress_bcs = []
        
        self.all_bcs = mem_velocity_bcs + bend_velocity_bcs + bend_stress_bcs

        self.mem_velocity_old, self.mem_stress_old, \
        self.bend_velocity_old, self.bend_stress_old = self.state_energy_old.subfunctions

        self.mem_velocity_new, self.mem_stress_new, \
        self.bend_velocity_new, self.bend_stress_new = self.state_energy_new.subfunctions

        # Set first value for the deformation gradient via Explicit Euler
        
        self.bend_displacement_old.assign(self.bend_displacement_old \
                                          + self.time_step/2*self.bend_velocity_old)

        self.time_displacement_old = fdrk.Constant(self.time_step/2)
        self.time_displacement_new = fdrk.Constant(self.time_step + self.time_step/2)
        self.actual_time_displacement = fdrk.Constant(self.time_step/2)

        # Set solver for the energy part
        self.states_energy_old = self.state_energy_old.subfunctions
        self.states_energy_new = self.state_energy_new.subfunctions

        self.energy_old = 0.5*mass_form_energy(self.states_energy_old, \
                                            self.states_energy_old, \
                                            problem.parameters, \
                                            membrane_inertia=membrane_inertia)
        
        self.energy_new = 0.5*mass_form_energy(self.states_energy_new, \
                                            self.states_energy_new, \
                                            self.problem.parameters, \
                                            membrane_inertia=membrane_inertia)

        self.a_form = operator_energy(self.time_step, \
                                tests_energy, \
                                trials_energy, \
                                self.bend_displacement_old, \
                                problem.parameters, \
                                problem.normal_versor, \
                                membrane_inertia=membrane_inertia, \
                                coupling = coupling)
        
        # # Spy matrix to look for error
        # A_petsc = fdrk.assemble(self.a_form, mat_type='aij').M.handle
        # A_scipy =  csr_matrix(A_petsc.getValuesCSR()[::-1])

        # plt.spy(A_scipy)
        # plt.show()

        self.l_form = functional_energy(self.time_step, \
                                tests_energy, \
                                self.states_energy_old, \
                                self.bend_displacement_old, \
                                problem.parameters, \
                                problem.normal_versor, \
                                membrane_inertia=membrane_inertia, \
                                coupling = coupling)

        linear_energy_problem = fdrk.LinearVariationalProblem(self.a_form, \
                                                            self.l_form, \
                                                            self.state_energy_new, \
                                                            bcs=self.all_bcs)
    
        self.linear_energy_solver = fdrk.LinearVariationalSolver(linear_energy_problem, \
                                                            solver_parameters=solver_parameters_energy)


    def integrate(self):

        # First the energy system is advanced at n+1
        self.linear_energy_solver.solve()

        # Compute solution for displacement at n+3∕2
        self.bend_displacement_new.assign(self.bend_displacement_old \
                                          + self.time_step * self.bend_velocity_new)

        self.actual_time_energy.assign(self.time_energy_new)
        self.actual_time_displacement.assign(self.time_displacement_new)


    def update_variables(self):
        self.state_energy_old.assign(self.state_energy_new)
        self.bend_displacement_old.assign(self.bend_displacement_new)

        self.time_energy_old.assign(self.actual_time_energy)
        self.time_energy_new.assign(float(self.time_energy_old) + self.time_step)

        self.time_displacement_old.assign(self.actual_time_displacement)
        self.time_displacement_new.assign(float(self.time_displacement_old) + self.time_step)


    def __str__(self):
        return "HamiltonianVonKarmanSolver"