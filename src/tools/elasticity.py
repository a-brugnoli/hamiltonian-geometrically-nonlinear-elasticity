import firedrake as fdrk
from src.tools.common import compute_min_max_mesh
import time
from firedrake.petsc import PETSc
import os 


def stiffness_tensor(strain, young_modulus, poisson_ratio):
    dim = strain.ufl_shape[0]

    stress = young_modulus/(1+poisson_ratio)*\
            (strain + poisson_ratio/(1-2*poisson_ratio)*fdrk.Identity(dim)*fdrk.tr(strain))

    return stress 


def compliance_tensor(stress, young_modulus, poisson_ratio):
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


def first_piola_neohookean(grad_disp, parameters, dim = 2):
    mu = parameters["mu"]
    lamda = parameters["lamda"]
    def_grad = fdrk.Identity(dim) + grad_disp
    inv_F_transpose = fdrk.inv(def_grad).T
    return mu*(def_grad - inv_F_transpose) + lamda * fdrk.ln(fdrk.det(def_grad)) * inv_F_transpose


def second_piola_neohookean(green_strain, parameters, dim = 2):
    mu = parameters["mu"]
    lamda = parameters["lamda"]
    inv_cauchy_strain = fdrk.inv(green_strain)
    return mu * (fdrk.Identity(dim) - inv_cauchy_strain) \
            + lamda/2 * fdrk.ln(fdrk.det(green_strain))*inv_cauchy_strain


def derivative_first_piola_neohookean(tensor, grad_disp, parameters, dim = 2):
    mu = parameters["mu"]
    lamda = parameters["lamda"]
    def_grad = fdrk.Identity(dim) + grad_disp
    invF = fdrk.inv(def_grad)
    inv_Ftr = fdrk.inv(def_grad).T

    return mu * tensor + (mu - lamda * fdrk.ln(fdrk.det(def_grad))) \
            * fdrk.dot(inv_Ftr, fdrk.dot(tensor.T, inv_Ftr)) \
            + lamda * fdrk.tr(fdrk.dot(invF, tensor)) * inv_Ftr


def natural_control_follower(test, displacement, traction_data_dict : dict):
    F = def_gradient(displacement)

    natural_control = 0 

    if traction_data_dict:

        for item in traction_data_dict.items():
            if item[0]=="follower":
                pass
            else:
                id = item[0]
                value_traction = item[1]


                if traction_data_dict["follower"]:
                    if id == "on_boundary":
                        natural_control +=fdrk.inner(test, fdrk.dot(F, value_traction))*fdrk.ds
                    else: 
                        natural_control +=fdrk.inner(test, fdrk.dot(F, value_traction))*fdrk.ds(id)
                else:
                    if id == "on_boundary":
                        natural_control +=fdrk.inner(test, value_traction)*fdrk.ds
                    else: 
                        natural_control +=fdrk.inner(test, value_traction)*fdrk.ds(id)

    return natural_control


def mass_form_energy(testfunctions, functions, params):
    
    young_modulus = params["E"]
    poisson_ratio = params["nu"]
    density = params["rho"]

    test_velocity, test_stress = testfunctions
    velocity, stress = functions

    linear_momentum = density * velocity
    strain = compliance_tensor(stress, young_modulus, poisson_ratio)

    return fdrk.inner(test_velocity, linear_momentum) * fdrk.dx + fdrk.inner(test_stress, strain)*fdrk.dx


def dynamics_form_energy(testfunctions, functions, displacement_midpoint):

    def_grad_midpoint = def_gradient(displacement_midpoint)

    test_velocity, test_stress = testfunctions
    velocity, stress = functions
    
    form =  - fdrk.inner(fdrk.grad(test_velocity), fdrk.dot(def_grad_midpoint, stress))*fdrk.dx \
            + fdrk.inner(test_stress, fdrk.dot(fdrk.transpose(def_grad_midpoint), fdrk.grad(velocity)))*fdrk.dx

    return form
    

def operator_energy(time_step, testfunctions, trialfunctions, displacement, parameters):
    """
    Construct operators arising from the implicit midpoint discretization of
    the energy part of the system

    A = M - 0.5 * dt *  J(d)
    """
    mass_operator = mass_form_energy(testfunctions, trialfunctions, parameters)
    dynamics_operator = dynamics_form_energy(testfunctions, trialfunctions, displacement)

    lhs_operator = mass_operator - 0.5 * time_step * dynamics_operator
    
    return lhs_operator
    

def functional_energy(time_step, testfunctions, old_states, displacement, control_midpoint_dict, params):
    """
    Construct functional arising from the implicit midpoint discretization of
    the energy part of the system
    b = ( M + 0.5 * dt * J(d_midpoint) ) x_old + B u_midpoint
    """

    mass_functional = mass_form_energy(testfunctions, old_states, params)
    dynamics_functional = dynamics_form_energy(testfunctions, old_states, displacement)

    natural_control = natural_control_follower(testfunctions[0], displacement, control_midpoint_dict)

    rhs_functional = mass_functional + 0.5 * time_step * dynamics_functional \
                                + time_step * natural_control
    
    return rhs_functional



def integrate(solver, T_end : float, title_file_displacement = None, \
              output_frequency = 10, collect_frames = True):
        
        time_vector = []
        time_vector.append(0)
        energy_vector = []
        energy_vector.append(fdrk.assemble(solver.energy_old))
        
        time_step_vec = []
        time_frames = []
        time_frames.append(0)
        list_mesh = []
        list_displacement = []

        list_min_max_coords = []
        if collect_frames:
            home_dir =os.environ['HOME']
            directory_largedata = f"{home_dir}/StoreResults/NonLinearElasticity/{str(solver)}/{str(solver.problem)}/"
            if not os.path.exists(directory_largedata):
                os.makedirs(directory_largedata, exist_ok=True)

            if title_file_displacement is None:

                path_file_displacement = f"{directory_largedata}/Displacement.pvd"
            else:
                if title_file_displacement.endswith(".pvd"):
                    path_file_displacement = directory_largedata + title_file_displacement
                else:
                    path_file_displacement = directory_largedata + title_file_displacement + ".pvd"

            outfile_displacement = fdrk.File(path_file_displacement)
            outfile_displacement.write(solver.displacement_old, time=0)

            displaced_mesh= solver.compute_displaced_mesh()
            list_mesh.append(displaced_mesh)

            list_displacement.append(solver.displacement_old.copy(deepcopy=True))

            displaced_coordinates_x = displaced_mesh.coordinates.dat.data[:, 0]
            displaced_coordinates_y = displaced_mesh.coordinates.dat.data[:, 1]
            displaced_coordinates_z = displaced_mesh.coordinates.dat.data[:, 2]

            min_max_coords_x = (min(displaced_coordinates_x), max(displaced_coordinates_x))
            min_max_coords_y = (min(displaced_coordinates_y), max(displaced_coordinates_y))
            min_max_coords_z = (min(displaced_coordinates_z), max(displaced_coordinates_z))

            list_min_max_coords = [min_max_coords_x, min_max_coords_y, min_max_coords_z]

        ii = 0
        actual_time = float(solver.actual_time)
        computing_time = 0

        while actual_time < T_end:

            start_iteration = time.time()
            solver.advance()
            end_iteration = time.time()
            elapsed_iteration = end_iteration - start_iteration
            computing_time += elapsed_iteration

            energy_vector.append(fdrk.assemble(solver.energy_new))
            time_step_vec.append(float(solver.time_step))

            solver.update_variables()

            actual_time = float(solver.actual_time)
            time_vector.append(actual_time)
            time_fraction = actual_time/T_end

            expected_total_computing_time = computing_time/time_fraction
            expected_remaining_time = expected_total_computing_time - computing_time
            ii+=1
            PETSc.Sys.Print(f"Iteration number {ii}. Actual time {actual_time:.3f}. Percentage : {time_fraction*100:.1f}%")
            PETSc.Sys.Print(f"Total computing time {computing_time:.1f}. Expected time to end : {expected_remaining_time/60:.1f} (min)")
            PETSc.Sys.Print(f'Actual time step {time_step_vec[-1]}')

            if ii % output_frequency and collect_frames:

                time_frames.append(actual_time)
                outfile_displacement.write(solver.displacement_old, time=actual_time)

                displaced_mesh = solver.compute_displaced_mesh()
                list_min_max_coords = compute_min_max_mesh(displaced_mesh, list_min_max_coords)
                list_mesh.append(displaced_mesh)
                list_displacement.append(solver.displacement_old.copy(deepcopy=True))


        dict_result = {"time": time_vector,
                    "energy": energy_vector, 
                    "time frames": time_frames, 
                    "time steps" : time_step_vec, 
                    "displacement mesh": list_mesh, 
                    "displacement solution": list_displacement, 
                    "minmax displacement": list_min_max_coords, 
                    "computing time": computing_time}
        
        
        return dict_result
