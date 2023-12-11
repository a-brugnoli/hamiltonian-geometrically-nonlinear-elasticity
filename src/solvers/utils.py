import firedrake as fdrk

def stiffness_tensor(strain, young_modulus, poisson_ratio):
    dim = strain.ufl_shape[0]

    stress = young_modulus/(1+poisson_ratio)*\
            (strain + poisson_ratio/(1-2*poisson_ratio)*fdrk.Identity(dim)*fdrk.tr(strain))

    return stress 


def compliance(stress, young_modulus, poisson_ratio):
    dim = stress.ufl_shape[0]
    # Compliance tensor for generic dimensions
    strain = 1 /(young_modulus) * ((1+poisson_ratio)*stress \
                                           - poisson_ratio * fdrk.Identity(dim) * fdrk.tr(stress))
    return strain


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


def mass_energy(testfunctions, functions, params):
    
    young_modulus = params["Young modulus"]
    poisson_ratio = params["Poisson ratio"]
    density = params["Density"]

    test_velocity, test_stress = testfunctions
    velocity, stress = functions

    linear_momentum = density * velocity
    strain = compliance(stress, young_modulus, poisson_ratio)

    return fdrk.inner(test_velocity, linear_momentum) * fdrk.dx + fdrk.inner(test_stress, strain)*fdrk.dx



def dynamics_energy(testfunctions, functions, displacement_midpoint):

    def_grad_midpoint = def_gradient(displacement_midpoint)

    test_velocity, test_stress = testfunctions
    velocity, stress = functions
    
    form =  - fdrk.inner(fdrk.grad(test_velocity), fdrk.dot(def_grad_midpoint, stress))*fdrk.dx \
                    + fdrk.inner(test_stress, fdrk.dot(fdrk.transpose(def_grad_midpoint), fdrk.grad(velocity)))*fdrk.dx

    return form
    

def operator_energy(time_step, testfunctions, trialfunctions, displacement, params):
    """
    Construct operators arising from the implicit midpoint discretization of
    the energy part of the system

    A = M - 0.5 * dt *  J(d)
    """
    mass_operator = mass_energy(testfunctions, trialfunctions, params)
    dynamics_operator = dynamics_energy(testfunctions, trialfunctions, displacement)

    lhs_operator = mass_operator - 0.5 * time_step * dynamics_operator
    
    return lhs_operator
    

def functional_energy(time_step, testfunctions, old_states, displacement, control_midpoint_dict, params):
    """
    Construct functional arising from the implicit midpoint discretization of
    the energy part of the system
    b = ( M + 0.5 * dt * J(d_midpoint) ) x_old + B u_midpoint
    """

    mass_functional = mass_energy(testfunctions, old_states, params)
    dynamics_functional = dynamics_energy(testfunctions, old_states, displacement)

    natural_control = natural_control_follower(testfunctions[0], displacement, control_midpoint_dict)

    rhs_functional = mass_functional + 0.5 * time_step * dynamics_functional \
                                + time_step * natural_control
    
    return rhs_functional