import firedrake as fdrk
from src.problems.problem import Problem
import numpy as np

class NonlinearLagrangianSolver:
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

        self.CG_vectorspace = fdrk.VectorFunctionSpace(problem.domain, "CG", pol_degree)

        # Spaces and functions
        self.displacement_old = fdrk.Function(self.CG_vectorspace)
        self.displacement_new = fdrk.Function(self.CG_vectorspace)

        self.green_lagrange_strain_old = green_lagrange_strain(self.displacement_old)
        self.green_lagrange_strain_new = green_lagrange_strain(self.displacement_new)

        self.second_piola_stress_old = stiffness_tensor(self.green_lagrange_strain_old, young_modulus, poisson_ratio)
        self.second_piola_stress_new = stiffness_tensor(self.green_lagrange_strain_new, young_modulus, poisson_ratio)

        self.first_piola_stress_old = first_piola_stress(self.displacement_old, young_modulus, poisson_ratio)
        self.first_piola_stress_new = first_piola_stress(self.displacement_new, young_modulus, poisson_ratio)

        self.velocity_old = fdrk.Function(self.CG_vectorspace)
        self.velocity_new = fdrk.Function(self.CG_vectorspace)

        self.acceleration_old = fdrk.Function(self.CG_vectorspace)    
        self.acceleration_new = fdrk.Function(self.CG_vectorspace)    

        self.displacement_midpoint = 0.5*(self.displacement_new + self.displacement_old)
        self.velocity_midpoint = 0.5*(self.velocity_new + self.velocity_old)
        self.acceleration_midpoint = 0.5*(self.acceleration_new + self.acceleration_old)

        self.green_lagrange_strain_midpoint = green_lagrange_strain(self.displacement_midpoint)
        self.second_piola_stress_midpoint = stiffness_tensor(self.green_lagrange_strain_midpoint, young_modulus, poisson_ratio)
        self.first_piola_stress_midpoint = first_piola_stress(self.displacement_midpoint, young_modulus, poisson_ratio)

        test_CG = fdrk.TestFunction(self.CG_vectorspace)
        trial_CG = fdrk.TrialFunction(self.CG_vectorspace)
        # Initial conditions and boundary conditions 
        dict_essential = problem.get_essential_bcs(self.time_new)
        disp_bc_data = dict_essential["displacement"]

        bcs_displacement = []
        bcs_acceleration_0 = []
        boundary_nodes = []
        for item in disp_bc_data.items():
            id_bc = item[0]
            value_bc = item[1]

            bc_item = fdrk.DirichletBC(self.CG_vectorspace, value_bc, id_bc)
            bcs_displacement.append(bc_item)
            bcs_acceleration_0.append(fdrk.DirichletBC(self.CG_vectorspace, fdrk.as_vector([0, 0]), id_bc))

            # for node in bc_item.nodes:
            #     boundary_nodes.append(node)

        dim_CG = self.CG_vectorspace.dim()
        list_dofs = list(range(dim_CG))

        expression_initial = problem.get_initial_conditions()
        displacement_t0 = expression_initial["displacement"]
        velocity_t0 = expression_initial["velocity"]

        self.displacement_old.assign(fdrk.interpolate(displacement_t0, self.CG_vectorspace))
        self.velocity_old.assign(fdrk.interpolate(velocity_t0, self.CG_vectorspace))
        
        # boundary_dofs_x = [element * 2 for element in boundary_nodes] 
        # boundary_dofs_y = [element * 2 + 1 for element in boundary_nodes] 
        # self.boundary_dofs = boundary_dofs_x + boundary_dofs_y
        # self.interior_dofs = list(set(list_dofs).difference(set(self.boundary_dofs)))

        # Set initial acceleration
        oper_acceleration = fdrk.inner(test_CG, density*trial_CG)*fdrk.dx

        traction_data_old = problem.get_natural_bcs(self.time_old)

        l_acceleration_0 = - fdrk.inner(fdrk.grad(test_CG), self.first_piola_stress_old)*fdrk.dx \
                           + natural_control_follower(test_CG, self.displacement_old, traction_data_old)
       
       
        fdrk.solve(oper_acceleration == l_acceleration_0, self.acceleration_old, bcs=bcs_acceleration_0)
        # Set non linear solver for the displacement

        self.beta = 1/4
        self.gamma = 1/2

        self.auxiliary_old = 1/(self.beta*self.time_step**2)*(self.displacement_old + self.time_step * self.velocity_old) \
                        + (1-2*self.beta)/(2*self.beta)*self.acceleration_old
        
        traction_data_new = problem.get_natural_bcs(self.time_new)

        res_displacement = 1/(self.beta*self.time_step**2)*fdrk.inner(test_CG, density*self.displacement_new)*fdrk.dx \
                         + fdrk.inner(fdrk.grad(test_CG), self.first_piola_stress_new)*fdrk.dx \
                         - fdrk.inner(test_CG, density*self.auxiliary_old)*fdrk.dx \
                         - natural_control_follower(test_CG, self.displacement_new, traction_data_new)
                    
        nonlinear_problem_displacement = fdrk.NonlinearVariationalProblem(res_displacement, 
                                                                        self.displacement_new, 
                                                                        bcs=bcs_displacement)
        
        
        self.displacement_solver = fdrk.NonlinearVariationalSolver(nonlinear_problem_displacement,
                                                                solver_parameters=solver_parameters)
    
        self.energy_old = 0.5 * fdrk.inner(self.velocity_old, density*self.velocity_old)*fdrk.dx \
                        + 0.5 * fdrk.inner(self.green_lagrange_strain_old, self.second_piola_stress_old)*fdrk.dx

        self.energy_new = 0.5 * fdrk.inner(self.velocity_new, density*self.velocity_new)*fdrk.dx \
                        + 0.5 * fdrk.inner(self.green_lagrange_strain_new, self.second_piola_stress_new)*fdrk.dx
        

    def integrate(self):
        self.displacement_solver.solve()
        self.acceleration_new.assign(1/(self.beta*self.time_step**2)*(self.displacement_new -self.displacement_old \
                                                                      - self.time_step * self.velocity_old) \
                                                                      - (1-2*self.beta)/(2*self.beta)*self.acceleration_old)
        self.velocity_new.assign(self.velocity_old 
            + self.time_step*((1-self.gamma)*self.acceleration_old + self.gamma*self.acceleration_new))
        
        self.actual_time.assign(self.time_new)
        
        # self.debug()

    
    def debug(self):
        assert fdrk.norm(1/self.time_step*(self.displacement_new-self.displacement_old) - self.velocity_midpoint)< 1e-9
        assert fdrk.norm(1/self.time_step*(self.velocity_new-self.velocity_old) - self.acceleration_midpoint)< 1e-9
        assert abs(fdrk.assemble(self.energy_new - self.energy_old)) < 1e-9
        

    def update_variables(self):
        self.displacement_old.assign(self.displacement_new)
        self.velocity_old.assign(self.velocity_new)
        self.acceleration_old.assign(self.acceleration_new)

        self.time_old.assign(self.actual_time)
        self.time_midpoint.assign(float(self.time_old) + self.time_step/2)
        self.time_new.assign(float(self.time_old) + self.time_step)


    def output_displaced_mesh(self):
        displaced_coordinates = fdrk.interpolate(self.problem.coordinates_mesh 
                                            + self.displacement_old, self.CG_vectorspace)

        return fdrk.Mesh(displaced_coordinates)
    

    def __str__(self):
        return "NonlinearLagrangianSolver"
    
    
def stiffness_tensor(strain, young_modulus, poisson_ratio):
    dim = strain.ufl_shape[0]

    stress = young_modulus/(1+poisson_ratio)*\
            (strain + poisson_ratio/(1-2*poisson_ratio)*fdrk.Identity(dim)*fdrk.tr(strain))

    return stress 


def def_gradient(vector):
    dim = vector.ufl_shape[0]
    return fdrk.Identity(dim) + fdrk.grad(vector)


def green_lagrange_strain(vector):
    return 1/2*(fdrk.grad(vector).T + fdrk.grad(vector) + fdrk.dot(fdrk.grad(vector).T, fdrk.grad(vector)))


def second_piola_stress(vector, young_modulus, poisson_ratio):
    return stiffness_tensor(green_lagrange_strain(vector), young_modulus, poisson_ratio)
    

def first_piola_stress(vector, young_modulus, poisson_ratio):
    return fdrk.dot(def_gradient(vector), second_piola_stress(vector, young_modulus, poisson_ratio))


def natural_control_follower(test, displacement, traction_data_dict : dict):
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





        




    