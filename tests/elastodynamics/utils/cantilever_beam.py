import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
from src.postprocessing.animators import animate_displacement
import matplotlib.pyplot as plt

def simulate_cantilever_beam(is_quad_mesh=False, linear=False, pol_degree=1, n_elem_x= 100):
    n_elem_y = int(n_elem_x/10)

    time_step = 1e-2
    T_end = 10
    n_time  = ceil(T_end/time_step)

    L_x = 100
    L_y = 10

    mesh = fdrk.RectangleMesh(n_elem_x, n_elem_y, L_x, L_y, quadrilateral=is_quad_mesh)

    coordinates_mesh = fdrk.SpatialCoordinate(mesh)
    x, y = coordinates_mesh
    h = 1

    density = fdrk.Constant(1)
    young_modulus = fdrk.Constant(1000)
    poisson_modulus = fdrk.Constant(0.3)

    def compliance(stress):
        eps_0 = 1 /(young_modulus * h) * ((1+poisson_modulus)*stress - poisson_modulus * fdrk.Identity(2) * fdrk.tr(stress))
        return eps_0


    def mass_defgradient(test_tensor, tensor):
        return fdrk.inner(test_tensor, tensor) * fdrk.dx


    def dyn_defgradient(test_tensor, vector):
        return fdrk.inner(test_tensor, fdrk.grad(vector)) * fdrk.dx


    def mass_energy_variables(test_vector, vector, test_symtensor, symtensor):
        return fdrk.inner(test_vector, density*vector)*fdrk.dx + fdrk.inner(test_symtensor, compliance(symtensor))*fdrk.dx


    def dyn_energy_variables(test_vector, vector, test_symtensor, symtensor, defgradient, linear=False):

        if linear:
            form = - fdrk.inner(fdrk.grad(test_vector), symtensor)*fdrk.dx \
                        + fdrk.inner(test_symtensor, fdrk.grad(vector))*fdrk.dx
        else:
            form = - fdrk.inner(fdrk.grad(test_vector), fdrk.dot(defgradient, symtensor))*fdrk.dx \
                        + fdrk.inner(test_symtensor, fdrk.dot(fdrk.transpose(defgradient), fdrk.grad(vector)))*fdrk.dx

        return form


    DG_tensorspace = fdrk.TensorFunctionSpace(mesh, "DG", pol_degree-1)

    CG_vectorspace = fdrk.VectorFunctionSpace(mesh, "CG", pol_degree)
    DG_symtensorspace = fdrk.TensorFunctionSpace(mesh, "DG", pol_degree-1, symmetry=True)

    space_energy = CG_vectorspace * DG_symtensorspace


    # Test and trial functions. Old, midpoint, new variables

    test_defgradient = fdrk.TestFunction(DG_tensorspace)
    trial_defgradient = fdrk.TrialFunction(DG_tensorspace)

    defgradient_old_half = fdrk.Function(DG_tensorspace)
    defgradient_new_half = fdrk.Function(DG_tensorspace)
    defgradient_midpoint_int = fdrk.Function(DG_tensorspace)

    test_velocity, test_stress = fdrk.TestFunctions(space_energy)
    trial_velocity, trial_stress = fdrk.TrialFunctions(space_energy)

    energy_var_old_int = fdrk.Function(space_energy)
    energy_var_midpoint_half = fdrk.Function(space_energy)
    energy_var_new_int = fdrk.Function(space_energy)

    velocity_old_int, stress_old_int = energy_var_old_int.subfunctions
    velocity_midpoint_half, stress_midpoint_half = energy_var_midpoint_half.subfunctions
    velocity_new_int, stress_new_int = energy_var_new_int.subfunctions

    displacement_old_int = fdrk.Function(CG_vectorspace)
    displacement_new_int = fdrk.Function(CG_vectorspace)

    # Simulation time

    time_old_defgradient = fdrk.Constant(0)
    time_midpoint_defgradient = fdrk.Constant(0)
    time_new_defgradient = fdrk.Constant(time_step/2)

    time_old_energy_var = fdrk.Constant(0)
    time_midpoint_energy_var = fdrk.Constant(time_step/2)
    time_new_energy_var = fdrk.Constant(time_step)

    # Operator, functional deformation gradient

    mass_operator_defgradient = mass_defgradient(test_defgradient, trial_defgradient)

    mass_functional_defgradient = mass_defgradient(test_defgradient, defgradient_old_half)
    dyn_functional_defgradient = dyn_defgradient(test_defgradient, velocity_old_int)

    b_functional_defgradient = mass_functional_defgradient + time_step * dyn_functional_defgradient

    problem_defgradient = fdrk.LinearVariationalProblem(a=mass_operator_defgradient,\
                                                        L=b_functional_defgradient, u=defgradient_new_half)

    solver_defgradient = fdrk.LinearVariationalSolver(problem=problem_defgradient, solver_parameters={'ksp_type': 'cg'})

    mass_operator_energy_var = mass_energy_variables(test_velocity, trial_velocity, \
                                                    test_stress, trial_stress)
    dyn_operator_energy_var = dyn_energy_variables(test_velocity, trial_velocity,\
                                                test_stress, trial_stress, defgradient_new_half, linear=linear)


    a_operator_energy_var = mass_operator_energy_var - 0.5*time_step*dyn_operator_energy_var


    mass_functional_energy_var = mass_energy_variables(test_velocity, velocity_old_int, \
                                                    test_stress, stress_old_int)
    dyn_functional_energy_var = dyn_energy_variables(test_velocity, velocity_old_int,\
                                                    test_stress, stress_old_int, defgradient_new_half, linear=linear)


    t_coutoff_forcing = fdrk.Constant(5)
    magnitude_traction = 50
    
    # traction_y = fdrk.sin(2*fdrk.pi*time_midpoint_energy_var/t_coutoff_forcing) *  \
    #     fdrk.conditional(fdrk.le(time_midpoint_energy_var, t_coutoff_forcing), magnitude_traction, 0)
    
    traction_y = time_midpoint_energy_var/t_coutoff_forcing *  \
        fdrk.conditional(fdrk.le(time_midpoint_energy_var, t_coutoff_forcing), magnitude_traction, 0)
    traction = fdrk.as_vector([fdrk.Constant(0), traction_y])

    boundary_form = fdrk.inner(test_velocity, fdrk.dot(defgradient_new_half, traction))*fdrk.ds(2)
    # boundary_form = fdrk.inner(test_velocity, traction)*fdrk.ds(2)

    b_functional_energy_var = mass_functional_energy_var + 0.5*time_step*dyn_functional_energy_var \
                                + time_step * boundary_form

    clamped_bc = fdrk.DirichletBC(space_energy.sub(0), fdrk.Constant((0, 0)), 1)

    problem_energy_var = fdrk.LinearVariationalProblem(a=a_operator_energy_var,\
                                                        L=b_functional_energy_var, \
                                                        u=energy_var_new_int, \
                                                        bcs=clamped_bc)


    solver_energ_var = fdrk.LinearVariationalSolver(problem=problem_energy_var)

    # Initial conditions

    defgradient_0 = fdrk.interpolate(fdrk.Constant(np.array([[1, 0], [0, 1]])), DG_tensorspace)
    defgradient_old_half.assign(defgradient_0)

    velocity_0 = fdrk.interpolate(fdrk.Constant(np.array([0, 0])), CG_vectorspace)
    velocity_old_int.assign(velocity_0)

    stress_0 = fdrk.interpolate(fdrk.Constant(np.array([[0, 0], [0, 0]])), DG_symtensorspace)
    stress_old_int.assign(stress_0)

    displacement_0 =  fdrk.interpolate(fdrk.Constant(np.array([0, 0])), CG_vectorspace)
    displacement_old_int.assign(displacement_0)


    #  energies

    energy_old_int = 0.5* mass_energy_variables(velocity_old_int, velocity_old_int, stress_old_int, stress_old_int)
    energy_new_int = 0.5 * mass_energy_variables(velocity_new_int, velocity_new_int, stress_new_int, stress_new_int)

    time_vector = np.linspace(0, T_end, num=n_time+1)
    energy_vector = np.zeros((n_time+1, ))
    energy_vector[0] = fdrk.assemble(energy_old_int)

    power_balance = time_step * fdrk.inner(velocity_midpoint_half, fdrk.dot(defgradient_new_half, traction))*fdrk.ds(2)
    # power_balance = time_step * fdrk.inner(velocity_midpoint_half, traction)*fdrk.ds(2)
    power_balance_vec = np.zeros((n_time, ))

    output_frequency = 10

    displaced_coordinates = fdrk.interpolate(coordinates_mesh + displacement_old_int, CG_vectorspace)

    displaced_mesh= fdrk.Mesh(displaced_coordinates)

    displaced_coordinates_x = displaced_mesh.coordinates.dat.data[:, 0]

    min_x_all = min(displaced_coordinates_x)
    max_x_all = max(displaced_coordinates_x)

    displaced_coordinates_y = displaced_mesh.coordinates.dat.data[:, 1]

    min_y_all = min(displaced_coordinates_y)
    max_y_all = max(displaced_coordinates_y)


    list_frames = []
    time_frames = []
    list_frames.append(displaced_mesh)
    time_frames.append(0)


    for ii in tqdm(range(1, n_time+1)):
        actual_time = ii*time_step
        # Solve the def gradient problem
        solver_defgradient.solve()
        defgradient_midpoint_int.assign(0.5*(defgradient_new_half+defgradient_old_half))

        if ii==1:
            defgradient_new_half.assign(defgradient_midpoint_int)

        # Solve the energy variables problem
        solver_energ_var.solve()
        energy_var_midpoint_half.assign(0.5*(energy_var_old_int + energy_var_new_int))
        velocity_midpoint_half.assign(0.5*(velocity_new_int + velocity_old_int))

        displacement_new_int.assign(displacement_old_int + time_step*velocity_midpoint_half)

        energy_vector[ii] = fdrk.assemble(energy_new_int)
        power_balance_vec[ii-1] = fdrk.assemble(power_balance)

        # New assign

        # time
        time_old_defgradient.assign(actual_time - time_step/2)
        time_midpoint_defgradient.assign(actual_time)
        time_new_defgradient.assign(actual_time + time_step/2)

        time_old_energy_var.assign(actual_time)
        time_midpoint_energy_var.assign(actual_time + time_step/2)
        time_new_energy_var.assign(actual_time + time_step)

        # variables
        defgradient_old_half.assign(defgradient_new_half)
        
        energy_var_old_int.assign(energy_var_new_int)
        displacement_old_int.assign(displacement_new_int)

        
        if ii % output_frequency == 0:

            displaced_coordinates = fdrk.interpolate(coordinates_mesh + displacement_old_int, CG_vectorspace)
            
            displaced_mesh = fdrk.Mesh(displaced_coordinates)

            displaced_coordinates_x = displaced_mesh.coordinates.dat.data[:, 0]

            min_x = min(displaced_coordinates_x)
            max_x = max(displaced_coordinates_x)

            if min_x<min_x_all:
                min_x_all = min_x
            if max_x>max_x_all:
                max_x_all = max_x

            displaced_coordinates_y = displaced_mesh.coordinates.dat.data[:, 1]

            min_y = min(displaced_coordinates_y)
            max_y = max(displaced_coordinates_y)
            
            if min_y<min_y_all:
                min_y_all = min_y
            if max_y>max_y_all:
                max_y_all = max_y


            list_frames.append(displaced_mesh)
            time_frames.append(actual_time)


    interval = 1e3 * output_frequency * time_step

    lim_x = (min_x_all, max_x_all)
    lim_y = (min_y_all, max_y_all)
    animation = animate_displacement(time_frames, list_frames, interval, \
                                               lim_x = lim_x, \
                                               lim_y = lim_y )

    animation.save(f"cantilever_linear_{linear}_quadmesh_{is_quad_mesh}.mp4", writer="ffmpeg")

    n_frames = len(list_frames)-1

    indexes_images = [int(n_frames/4), int(n_frames/2), int(3*n_frames/4), int(n_frames)]

    for kk in indexes_images:

        fig, axes = plt.subplots()
        axes.set_aspect("equal")
        fdrk.triplot(list_frames[kk], axes=axes)
        # axes.set_title(f"Displacement at time $t={time_image}$" + r"$[\mathrm{s}]$", loc='center')
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_xlim(lim_x)
        axes.set_ylim(lim_y)

        plt.savefig(f"Displacement_linear_{linear}_index_{kk}.eps", bbox_inches='tight', dpi='figure', format='eps')


    return time_vector, energy_vector, power_balance_vec