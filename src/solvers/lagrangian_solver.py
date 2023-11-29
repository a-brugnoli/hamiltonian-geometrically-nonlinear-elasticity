import firedrake as fdrk
from src.problems.problem import Problem


class LagrangianNonLinearSolver:
    def __init__(self,
                 problem: Problem,
                 time_step: float,
                 pol_degree= 1,
                 solver_parameters= {}):
        
        # Set times 
        self.problem = problem
        self.time_step = time_step
        self.time_old = fdrk.Constant(0)
        self.time_midpoint = fdrk.Constant(self.time_step/2)
        self.time_new = fdrk.Constant(self.time_step)
        self.actual_time = fdrk.Constant(0)

        # Physical parameters
        density = problem.parameters["Density"]
        young_modulus = problem.parameters["Young modulus"]
        poisson_ratio = problem.parameters["Poisson ratio"]

        CG_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)

        self.space_displacement = CG_vectorspace

        # Spaces and functions
        self.displacement_old = fdrk.Function(CG_vectorspace)
        self.displacement_new = fdrk.Function(CG_vectorspace)

        self.velocity_old = fdrk.Function(CG_vectorspace)
        self.velocity_new = fdrk.Function(CG_vectorspace)

        self.acceleration_old = fdrk.Function(CG_vectorspace)    
        self.acceleration_new = fdrk.Function(CG_vectorspace)    

        test_CG = fdrk.TestFunction(CG_vectorspace)
        trial_CG = fdrk.TrialFunction(CG_vectorspace)

        # Initial conditions and boundary conditions 
        expression_initial = problem.get_initial_conditions()
        displacement_t0 = expression_initial["displacement"]
        velocity_t0 = expression_initial["velocity"]

        self.displacement_old.assign(fdrk.interpolate(displacement_t0, CG_vectorspace))
        self.velocity_old.assign(fdrk.interpolate(velocity_t0, CG_vectorspace))

        dict_essential = problem.get_essential_bcs(self.time_new)
        disp_bc_data = dict_essential["displacement"]

        bcs_displacement = []
        for item in disp_bc_data.items():
            id_bc = item[0]
            value_bc = item[1]
            bcs_displacement.append(fdrk.DirichletBC(self.space_displacement, value_bc, id_bc))



        # Set initial acceleration
        oper_acceleration = mass_form(test_CG, trial_CG, density)

        traction_data_old = problem.get_natural_bcs(self.time_old)
        weak_piola_old = weak_div_piola(test_CG, 
                                        self.displacement_old, 
                                        traction_data_old, 
                                        young_modulus, 
                                        poisson_ratio)

        
        fdrk.solve(oper_acceleration == weak_piola_old, self.acceleration_old)
        # Set non linear solver for the displacement
        # Energy preserving Newmark scheme

        self.beta = 1/4
        self.gamma = 1/2

        auxiliary_old = 1/( self.beta*time_step**2)*(self.displacement_old + time_step * self.velocity_old) \
                        + (1-2* self.beta)/(2* self.beta)*self.acceleration_old
        
        traction_data_new = problem.get_natural_bcs(self.time_new)
        weak_piola_new = weak_div_piola(test_CG, 
                                        self.displacement_new, 
                                        traction_data_new, 
                                        young_modulus, 
                                        poisson_ratio)


        residual_displacement = 1/( self.beta*time_step**2)*mass_form(test_CG, self.displacement_new, density) \
                                - weak_piola_new - mass_form(test_CG, auxiliary_old, density)

        nonlinear_problem_displacement = fdrk.NonlinearVariationalProblem(residual_displacement, 
                                                                          self.displacement_new, 
                                                                          bcs_displacement)
        
        self.displacement_solver = fdrk.NonlinearVariationalSolver(nonlinear_problem_displacement,
                                                                   solver_parameters=solver_parameters)
        
        acceleration_problem = fdrk.LinearVariationalProblem(oper_acceleration, 
                                                             weak_piola_new, 
                                                             self.acceleration_new)
        
        self.acceleration_solver =  fdrk.LinearVariationalSolver(acceleration_problem, 
                                                            solver_parameters={"ksp_type":"cg"})
        
        self.energy_old = 0.5 * mass_form(self.velocity_old, self.velocity_old, density) \
                        + 0.5 * fdrk.inner(green_lagrange_strain(self.displacement_old), 
                        stiffness(green_lagrange_strain(self.displacement_old), young_modulus, poisson_ratio)) * fdrk.dx

        self.energy_new = 0.5 * mass_form(self.velocity_new, self.velocity_new, density) \
                        + 0.5 * fdrk.inner(green_lagrange_strain(self.displacement_new), 
                        stiffness(green_lagrange_strain(self.displacement_new), young_modulus, poisson_ratio)) * fdrk.dx


    def integrate(self):
        self.displacement_solver.solve()

        self.acceleration_solver.solve()

        self.velocity_new.assign(self.velocity_old 
            + self.time_step*((1+self.gamma)*self.acceleration_old + self.gamma*self.acceleration_new))

        self.actual_time.assign(self.time_new)


    def update_variables(self):
        self.displacement_old.assign(self.displacement_new)
        self.velocity_old.assign(self.velocity_new)
        self.acceleration_old.assign(self.acceleration_new)

        self.time_old.assign(self.actual_time)
        self.time_midpoint.assign(float(self.time_old) + self.time_step/2)
        self.time_new.assign(float(self.time_old) + self.time_step)


    def __str__(self):
        return "LagrangianSolver"
    
    
def stiffness(strain, young_modulus, poisson_ratio):
    dim = strain.ufl_shape[0]

    stress = young_modulus/(1+poisson_ratio)*\
            (strain + poisson_ratio/(1-2*poisson_ratio)*fdrk.Identity(dim)*fdrk.tr(strain))

    return stress 


def mass_form(test, function, density):
    return fdrk.inner(test, density*function)*fdrk.dx


def def_gradient(displacement):
    dim = displacement.ufl_shape[0]

    return fdrk.Identity(dim) + fdrk.grad(displacement)


def green_lagrange_strain(displacement):
    dim = displacement.ufl_shape[0]
    
    F = def_gradient(displacement)

    green_lagrange_strain = 1/2*(fdrk.dot(F.T, F) - fdrk.Identity(dim))

    return green_lagrange_strain


def first_Piola(displacement, young_modulus, poisson_ratio):
    F = def_gradient(displacement)

    second_Piola = stiffness(green_lagrange_strain(displacement), young_modulus, poisson_ratio)

    return fdrk.dot(F, second_Piola)


def natural_control(test, displacement, traction_data_dict : dict):
    F = def_gradient(displacement)
    for item in traction_data_dict.items():
        id = item[0]
        value_traction = item[1]

        natural_control = 0 

        if id == "on_boundary":
            natural_control +=fdrk.inner(test, fdrk.dot(F, value_traction))*fdrk.ds
        else: 
            natural_control +=fdrk.inner(test, fdrk.dot(F, value_traction))*fdrk.ds(id)

    return natural_control


def weak_div_piola(test, displacement, traction_data_dict : dict, young_modulus, poisson_ratio):

    return -fdrk.inner(fdrk.grad(test), first_Piola(displacement, young_modulus, poisson_ratio))*fdrk.dx \
           + natural_control(test, displacement, traction_data_dict)
        




    