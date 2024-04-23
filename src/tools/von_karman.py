import firedrake as fdrk
from firedrake.petsc import PETSc


def sym_grad(vector):
    return fdrk.sym(fdrk.grad(vector))


def membrane_stiffness(membrane_strain, parameters):
    thickness = parameters["h"]
    young_modulus = parameters["E"]
    poisson_ratio = parameters["nu"]

    D_men = fdrk.Constant(young_modulus * thickness / (1 - poisson_ratio ** 2))

    membrane_stress = D_men * ((1 - poisson_ratio) * membrane_strain \
                     + poisson_ratio * fdrk.Identity(2) * fdrk.tr(membrane_strain))
    
    return membrane_stress


def membrane_compliance(membrane_stress, parameters):
    thickness = parameters["h"]
    young_modulus = parameters["E"]
    poisson_ratio = parameters["nu"]

    C_men = fdrk.Constant(1 / (young_modulus * thickness))

    membrane_strain = C_men * ((1 + poisson_ratio) * membrane_stress \
                        - poisson_ratio * fdrk.Identity(2) * fdrk.tr(membrane_stress))

    return membrane_strain


def bending_stiffness(bending_strain, parameters):
    thickness = parameters["h"]
    young_modulus = parameters["E"]
    poisson_ratio = parameters["nu"]

    D_bend = fdrk.Constant(young_modulus * thickness ** 3 / (1 - poisson_ratio ** 2) / 12)

    bending_stress = D_bend * ((1 - poisson_ratio) * bending_strain \
                               + poisson_ratio * fdrk.Identity(2) * fdrk.tr(bending_strain))
    
    return bending_stress


def bending_compliance(bending_stress, parameters):
    thickness = parameters["h"]
    young_modulus = parameters["E"]
    poisson_ratio = parameters["nu"]

    C_bend = fdrk.Constant(12 / (young_modulus * thickness ** 3))

    bending_strain = C_bend * ((1+ poisson_ratio)*bending_stress 
                     - poisson_ratio * fdrk.Identity(2) * fdrk.tr(bending_stress))
    
    return bending_strain


def mass_form_energy(testfunctions, functions, parameters, membrane_inertia=True):
    density = parameters["rho"]
    thickness = parameters["h"]

    test_mem_velocity, test_mem_stress, test_bend_velocity, test_bend_stress = testfunctions
    mem_velocity, mem_stress, bend_velocity, bend_stress = functions

    mem_momentum = density * thickness * mem_velocity
    bend_momentum = density * thickness * bend_velocity

    mem_strain = membrane_compliance(mem_stress, parameters)
    bend_strain = bending_compliance(bend_stress, parameters)

    mass_form = fdrk.inner(test_mem_stress, mem_strain) * fdrk.dx + \
                fdrk.inner(test_bend_velocity, bend_momentum) * fdrk.dx + \
                fdrk.inner(test_bend_stress, bend_strain) * fdrk.dx
    
    if membrane_inertia:
        mass_form+= fdrk.inner(test_mem_velocity, mem_momentum) * fdrk.dx 

    return mass_form



def dynamics_form_energy(testfunctions, functions, vert_displacement, normal, coupling=True):
    
    test_mem_velocity, test_mem_stress, test_bend_velocity, test_bend_stress = testfunctions
    mem_velocity, mem_stress, bend_velocity, bend_stress = functions

    dynamics_membrane = - fdrk.inner(fdrk.sym(fdrk.grad(test_mem_velocity)), mem_stress) * fdrk.dx \
                        + fdrk.inner(test_mem_stress, fdrk.sym(fdrk.grad(mem_velocity))) * fdrk.dx 
            
    dynamics_bending = - fdrk.inner(fdrk.grad(fdrk.grad(test_bend_velocity)), bend_stress) * fdrk.dx \
    + fdrk.jump(fdrk.grad(test_bend_velocity), normal) * fdrk.dot(fdrk.dot(bend_stress('+'), normal('+')), normal('+')) * fdrk.dS \
    + fdrk.dot(fdrk.grad(test_bend_velocity), normal) * fdrk.dot(fdrk.dot(bend_stress, normal), normal) * fdrk.ds \
    + fdrk.inner(test_bend_stress, fdrk.grad(fdrk.grad(bend_velocity))) * fdrk.dx \
    - fdrk.dot(fdrk.dot(test_bend_stress('+'), normal('+')), normal('+')) * fdrk.jump(fdrk.grad(bend_velocity), normal) * fdrk.dS \
    - fdrk.dot(fdrk.dot(test_bend_stress, normal), normal) * fdrk.dot(fdrk.grad(bend_velocity), normal) * fdrk.ds
    
    dynamics_form = dynamics_membrane + dynamics_bending 

    if coupling:
        dynamics_coupling = fdrk.inner(test_mem_stress, fdrk.sym(fdrk.outer(fdrk.grad(vert_displacement), fdrk.grad(bend_velocity)))) * fdrk.dx \
        - fdrk.inner(fdrk.sym(fdrk.outer(fdrk.grad(test_bend_velocity), fdrk.grad(vert_displacement))), mem_stress) * fdrk.dx

        dynamics_form += dynamics_coupling
    
    return dynamics_form 
    

def operator_energy(time_step, testfunctions, trialfunctions,\
                    displacement, parameters, normal, \
                    membrane_inertia=True, coupling=True):
    """
    Construct operators arising from the implicit midpoint discretization of
    the energy part of the system

    A = M - 0.5 * dt *  J(d)
    """
    mass_operator = mass_form_energy(testfunctions, trialfunctions, parameters, membrane_inertia)
    dynamics_operator = dynamics_form_energy(testfunctions, trialfunctions,\
                                            displacement, normal, coupling)

    lhs_operator = mass_operator - 0.5 * time_step * dynamics_operator
    
    return lhs_operator
    

def functional_energy(time_step, testfunctions, old_states,\
                    displacement, parameters, normal, \
                    membrane_inertia=True, coupling=True):
    """
    Construct functional arising from the implicit midpoint discretization of
    the energy part of the system
    b = ( M + 0.5 * dt * J(d) ) x_old
    """
    mass_functional = mass_form_energy(testfunctions, old_states, parameters, membrane_inertia)
    dynamics_functional = dynamics_form_energy(testfunctions, old_states,\
                                            displacement, normal, coupling)

    rhs_functional = mass_functional + 0.5 * time_step * dynamics_functional
    
    return rhs_functional



if __name__=="__main__":

    parameters = {"E": 2*10**11, "h": 0.001, "nu": 0.3}

    mesh = fdrk.UnitSquareMesh(10, 10)
    tensor_exp = fdrk.Constant(((2.5, 1.3), (1.3, 3.7)))

    tensor_space = fdrk.TensorFunctionSpace(mesh, "CG", 2, symmetry=True)
    tensor_function = fdrk.interpolate(tensor_exp, tensor_space)

    tol = 1e-12

    tensor_function_mem = membrane_compliance(membrane_stiffness(tensor_function, parameters), parameters)
    tensor_function_bend = bending_compliance(bending_stiffness(tensor_function, parameters), parameters)
    diff_membrane = tensor_function_mem - tensor_function
    diff_bending = tensor_function_bend - tensor_function

    print(fdrk.norm(diff_membrane), fdrk.norm(diff_bending))
    assert fdrk.norm(diff_membrane) < tol
    assert fdrk.norm(diff_bending) < tol
    
