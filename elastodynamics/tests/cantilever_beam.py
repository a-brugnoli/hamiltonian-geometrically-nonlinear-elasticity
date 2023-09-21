import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt

pol_degree = 1
n_elem_x = 10
n_elem_y = 3

time_step = 0.001
T_end = 2000*time_step
n_time  = ceil(T_end/time_step)

L_x = 100
L_y = 5
mesh = fdrk.RectangleMesh(n_elem_x, n_elem_y, L_x, L_y)

h = 0.1

density = fdrk.Constant(2700)
young_modulus = fdrk.Constant(70 * 10**3)
poisson_modulus = fdrk.Constant(0.3)

def compliance(stress):
    eps_0 = 1 /(young_modulus * h) * ((1+poisson_modulus)*stress- poisson_modulus * fdrk.Identity(2) * fdrk.tr(stress))
    return eps_0


DG_tensorspace = fdrk.TensorFunctionSpace(mesh, "DG", pol_degree-1)
CG_vectorspace = fdrk.VectorFunctionSpace(mesh, "CG", pol_degree)
DG_symtensorspace = fdrk.TensorFunctionSpace(mesh, "DG", pol_degree-1)

space_energy = CG_vectorspace * DG_symtensorspace


def mass_defgradient(test_tensor, tensor):
    return fdrk.inner(test_tensor, tensor) * fdrk.dx


def dyn_defgradient(test_tensor, vector):
    return fdrk.inner(test_tensor, fdrk.grad(vector)) * fdrk.dx


def mass_energy_variables(test_vector, vector, test_symtensor, symtensor):
    return fdrk.inner(test_vector, density*vector)*fdrk.dx + fdrk.inner(test_symtensor, compliance(symtensor))*fdrk.dx


def dyn_energy_variables(test_vector, vector, test_symtensor, symtensor, defgradient):

    return - fdrk.inner(fdrk.grad(test_vector), fdrk.dot(defgradient, symtensor))*fdrk.dx \
           + fdrk.inner(test_symtensor, fdrk.dot(fdrk.transpose(defgradient), fdrk.grad(vector)))*fdrk.dx



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
velocity_midpoint_int, stress_midpoint_int = energy_var_midpoint_half.subfunctions
velocity_new_int, stress_new_int = energy_var_new_int.subfunctions

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
                                               test_stress, trial_stress, defgradient_new_half)


a_operator_energy_var = mass_operator_energy_var - 0.5*time_step*dyn_operator_energy_var


mass_functional_energy_var = mass_energy_variables(test_velocity, velocity_old_int, \
                                                   test_stress, stress_old_int)
dyn_functional_energy_var = dyn_energy_variables(test_velocity, velocity_old_int,\
                                                test_stress, stress_old_int, defgradient_new_half)

traction_y = fdrk.conditional(fdrk.le(time_midpoint_energy_var, 0.2), 1, 0)
traction = fdrk.as_vector([fdrk.Constant(0), traction_y])
boundary_form = fdrk.inner(test_velocity, traction)*fdrk.ds(2)


b_functional_energy_var = mass_functional_energy_var + 0.5*time_step*dyn_functional_energy_var \
                            + time_step * boundary_form


clamped_bc = fdrk.DirichletBC(space_energy.sub(0), fdrk.Constant((0, 0)), 1)

problem_energy_var = fdrk.LinearVariationalProblem(a=mass_operator_energy_var,\
                                                    L=b_functional_energy_var, u=energy_var_new_int)

solver_energ_var = fdrk.LinearVariationalSolver(problem=problem_energy_var)



# Initial conditions

defgradient_0 = fdrk.interpolate(fdrk.Constant(np.array([[0, 0], [0, 0]])), DG_tensorspace)
defgradient_old_half.assign(defgradient_0)

velocity_0 = fdrk.interpolate(fdrk.Constant(np.array([0, 0])), CG_vectorspace)
velocity_old_int.assign(velocity_0)

stress_0 = fdrk.interpolate(fdrk.Constant(np.array([[0, 0], [0, 0]])), DG_symtensorspace)
stress_old_int.assign(stress_0)


#  energies

energy_old_int = mass_energy_variables(velocity_old_int, velocity_old_int, stress_old_int, stress_old_int)
energy_new_int = mass_energy_variables(velocity_new_int, velocity_new_int, stress_new_int, stress_new_int)

time_vector = np.linspace(0, T_end, num=n_time+1)
energy_vector = np.zeros((n_time+1, ))
energy_vector[0] = fdrk.assemble(energy_old_int)

for ii in tqdm(range(1, n_time+1)):
    actual_time = ii*time_step
    # Solve the def gradient problem

    if ii==1:
        solver_defgradient.solve()
        defgradient_midpoint_int.assign(0.5*(defgradient_new_half+defgradient_old_half))
        defgradient_new_half.assign(defgradient_midpoint_int)
    if ii>1:
        solver_defgradient.solve()
        defgradient_midpoint_int.assign(0.5*(defgradient_new_half+defgradient_old_half))

    solver_energ_var.solve()
    energy_var_midpoint_half.assign(0.5*(energy_var_old_int + energy_var_new_int))



    # New assign

    time_old_defgradient.assign(actual_time - time_step/2)
    time_midpoint_defgradient.assign(actual_time)
    time_new_defgradient.assign(actual_time + time_step/2)

    time_old_energy_var.assign(actual_time)
    time_midpoint_energy_var.assign(actual_time + time_step/2)
    time_new_energy_var.assign(actual_time + time_step)

    defgradient_old_half.assign(defgradient_new_half)
    energy_var_old_int.assign(energy_var_new_int)

    energy_vector[ii] = fdrk.assemble(energy_old_int)

plt.plot(energy_vector)
plt.show()


fdrk.quiver(velocity_new_int)
plt.show()





