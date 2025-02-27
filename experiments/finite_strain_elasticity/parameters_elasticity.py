import numpy as np
from math import pi
from experiments.finite_strain_elasticity.finite_strain_elasticity import FiniteStrainElasticity
import os 

# Mechanical parameters
rho = 1.1*10**3 # Density [kg/m^3]
E = 17*10**6 # Young modulus [Pa]
nu = 0.3

mu = E / (2*(1 + nu))
lamda = E*nu/((1 - 2*nu)*(1 + nu))
kappa = lamda + 2/3*mu

# Geometrical parameters
Lx, Ly, Lz = 1, 1, 6
n_elements = (6, 6, 36)
n_elements_x, n_elements_y, n_elements_z = n_elements
mesh_size_x = Lx/n_elements_x
mesh_size_y = Ly/n_elements_y
mesh_size_z = Lz/n_elements_z
mesh_size = min(mesh_size_x, mesh_size_y, mesh_size_z)

# Wave speed and connection with time step
wave_speed = np.sqrt((kappa + 4/3*mu)/rho)    

# To have a stable leapfrog method, this needs to be chosen to 4
sec_coeff = 1

# CFL in finite differences scheme (or finite element with mass lumping)
dt_max = mesh_size/wave_speed
dt_CFL = dt_max/sec_coeff

# # Time step to have sufficient resolution
dt_base = dt_CFL
t_end_approx = 1/2
n_steps_approx = np.round(t_end_approx/dt_base).astype(int)

# Computation of the final time and number of steps to collect a maximum number of 
# output data set to 1e4
max_output = 10000
if n_steps_approx+1<=max_output:
    n_steps = n_steps_approx
    n_sim_output = n_steps+1
else:
    n_sim_output = max_output
    n_steps_output = n_sim_output-1
    n_steps = np.round(n_steps_approx/n_steps_output)*n_steps_output

t_end = n_steps.astype(int)*dt_base
t_span = [0, t_end]

# Declaration of the model for the finest simulation
coeff_reference = 2**7
dt_reference = dt_base/coeff_reference

bending_column = FiniteStrainElasticity(time_step=dt_base, t_span=t_span, n_output= n_sim_output,
                                        n_elem = n_elements, Lx=Lx, Ly=Ly, Lz=Lz, 
                                        rho = rho, E = E, nu = nu)
n_dofs_disp = bending_column.n_dofs_disp
space_dim = bending_column.domain.topological_dimension()
# Point to analyze results
x_point = np.array([0, 0, Lz])
array_coordinates = bending_column.domain.coordinates.dat.data
distances = np.sum((array_coordinates - x_point)**2, axis=1)
# Find index of minimum distance
index_point = np.argmin(distances)


t_vec_output = bending_column.t_vec_output
t_vec_output_ms = t_vec_output*1e3
dt_output = np.mean(np.diff(t_vec_output)) 

n_cases = 5
time_step_vec = np.array([dt_base/2**n for n in range(n_cases)])
time_step_vec_mus = time_step_vec*1e5

mask_stable_leapfrog = time_step_vec <= dt_CFL/4
n_cases_stable_leapfrog = np.sum(mask_stable_leapfrog==True)
time_step_stable_leapfrog = time_step_vec[mask_stable_leapfrog]


# Paths for results
home_dir = os.path.expanduser("~")
directory_results = home_dir + "/StoreResults/FiniteStrainElasticity/results/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)

file_time = directory_results + "time_vector.pkl"
file_results_reference = directory_results + "results_reference.pkl"
file_results_leapfrog = directory_results + "results_leapfrog.pkl"
file_results_dis_gradient = directory_results + "results_discrete_gradient.pkl"
file_results_lin_implicit = directory_results + "results_linear_implicit.pkl"

directory_images = f"{os.path.dirname(os.path.abspath(__file__))}/images/"
if not os.path.exists(directory_images):
    os.makedirs(directory_images)

# Paraview folder
paraview_directory = home_dir + "/StoreResults/FiniteStrainElasticity/"

