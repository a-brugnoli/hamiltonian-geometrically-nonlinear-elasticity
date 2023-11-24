import firedrake as fdrk
from .system_operators import DynamicalOperators
from src.problems.problem import Problem

class Elastodynamics(DynamicalOperators):

    def __init__(self, problem: Problem, pol_degree: int):
        super().__init__(problem, pol_degree)


    def _set_space(self):
        DG_tensorspace = fdrk.TensorFunctionSpace(self.domain, "DG", self.pol_degree-1)

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", self.pol_degree)
        DG_symtensorspace = fdrk.TensorFunctionSpace(self.domain, "DG", self.pol_degree-1, symmetry=True)

        self.space_def_gradient = DG_tensorspace
        self.space_energy = CG_vectorspace * DG_symtensorspace


    def interpolated_initial_conditions(self, expression_initial: dict):

        def_gradient_exp = expression_initial["def_gradient"]
        velocity_exp = expression_initial["velocity"]
        stress_exp = expression_initial["stress"]
        displacement_exp = expression_initial["displacement"]

         # interpolation of the expression on their function spaces
        def_gradient = fdrk.interpolate(def_gradient_exp, self.space_def_gradient)
        velocity = fdrk.interpolate(velocity_exp, self.space_energy.sub(0))
        stress = fdrk.interpolate(stress_exp, self.space_energy.sub(1))

        displacement = fdrk.interpolate(displacement_exp, self.space_energy.sub(0))

        return (def_gradient, velocity, stress, displacement)
    

    def boundary_conditions(self, time_ess: fdrk.Constant, time_nat: fdrk.Constant):
        bc_dictionary = self.problem.get_boundary_conditions(time_ess, time_nat)

        essential_bc_data = bc_dictionary["velocity"]

        natural_bcs_dict = bc_dictionary["traction"]

        essential_bcs_list = []

        for item in essential_bc_data.items():
            id_bc = item[0]
            value_bc = item[1]
            essential_bcs_list.append(fdrk.DirichletBC(self.space_energy.sub(0), value_bc, id_bc))

        return essential_bcs_list, natural_bcs_dict
    

    def control(self, testfunctions, def_gradient_midpoint, controls_dict):
        test_velocity = testfunctions[0]
        natural_control = 0 

        for item in controls_dict.items():
            id = item[0]
            value_traction = item[1]

            if id == "on_boundary":
                natural_control +=fdrk.inner(test_velocity, fdrk.dot(def_gradient_midpoint, value_traction))*fdrk.ds
            else: 
                natural_control +=fdrk.inner(test_velocity, fdrk.dot(def_gradient_midpoint, value_traction))*fdrk.ds(id)

        return natural_control
    

    def mass_def_gradient(self, test_def_gradient, def_gradient):
        return fdrk.inner(test_def_gradient, def_gradient) * fdrk.dx


    def dynamics_def_gradient(self, test_def_gradient, velocity):
        return fdrk.inner(test_def_gradient, fdrk.grad(velocity)) * fdrk.dx
    

    def mass_energy(self, testfunctions, functions):
        params = self.problem.parameters

        density = params["Density"]
        young_modulus = params["Young modulus"]
        poisson_modulus = params["Poisson modulus"]
        h = params["Thickness"]

        test_velocity, test_stress = testfunctions
        velocity, stress = functions

        linear_momentum = density * velocity
        strain = 1 /(young_modulus * h) * ((1+poisson_modulus)*stress \
                                           - poisson_modulus * fdrk.Identity(2) * fdrk.tr(stress))

        return fdrk.inner(test_velocity, linear_momentum) * fdrk.dx + fdrk.inner(test_stress, strain)*fdrk.dx
    

    def dynamics_energy(self, testfunctions, functions, def_grad_midpoint):

        test_velocity, test_stress = testfunctions
        velocity, stress = functions
        
        form =  - fdrk.inner(fdrk.grad(test_velocity), fdrk.dot(def_grad_midpoint, stress))*fdrk.dx \
                        + fdrk.inner(test_stress, fdrk.dot(fdrk.transpose(def_grad_midpoint), fdrk.grad(velocity)))*fdrk.dx

        return form
    


