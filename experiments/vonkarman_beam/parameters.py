import numpy as np
from math import pi
from experiments.vonkarman_beam.vonkarman_beam import VonKarmanBeam
import os 

# Mechanical parameters
rho = 2700 # Density [kg/m^3]
E = 70*10**9 # Young modulus [Pa]

# Geometrical parameters
width = 0.002 # m
height = 0.002 # m
A = width*height # Cross section
I = (width * height**3) / 12 # Second moment of inertia
L = 1

n_elements = 50
mesh_size = L/n_elements

# # Initial condition
alpha = .1
ampl_hor_disp_0 = alpha*height
ampl_ver_disp_0 = alpha*height

# Wave speed and connection with time step
wave_speed_bending = np.sqrt(E*I/(rho*A))
wave_speed_traction = np.sqrt(E/rho)

# # In finite elements, the minimum security coefficient is:
# - 4 for bending 
# - 2 for traction
sec_coeff_bend = 4
sec_coeff_traction = 2

# CFL in finite differences scheme (or finite element with mass lumping)
dt_CFL_bending = mesh_size**2/(2*wave_speed_bending)
dt_CFL_traction = mesh_size/wave_speed_traction

dt_CFL_bending_cons = dt_CFL_bending/sec_coeff_bend
dt_CFL_traction_cons = dt_CFL_traction/sec_coeff_traction

# # Time step to have sufficient resolution
dt_base = dt_CFL_bending_cons

omega1_bending = (pi/L)**2*wave_speed_bending
T1_bending = 2*pi/omega1_bending

t_end_approx = 0.1*T1_bending
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

beam = VonKarmanBeam(time_step=dt_reference, t_span=t_span, n_output= n_sim_output,\
                        n_elem = n_elements, q0_hor = ampl_hor_disp_0, q0_ver=ampl_ver_disp_0, \
                        rho = rho, E = E, I = I, A=A, L=L)

x_vec = beam.x_vec
x_point = L/4
index_point = np.argmin(np.abs(x_vec - x_point))


t_vec_output = beam.t_vec_output
t_vec_output_ms = t_vec_output*1e3
dt_output = np.mean(np.diff(t_vec_output)) 

# Paths for results
directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)

file_results_reference = directory_results + "results_reference.pkl"
file_results_leapfrog = directory_results + "results_leapfrog.pkl"
file_results_dis_gradient = directory_results + "results_discrete_gradient.pkl"
file_results_lin_implicit = directory_results + "results_linear_implicit.pkl"

n_cases = 6
time_step_vec = np.array([dt_base/2**n for n in range(n_cases)])
time_step_vec_mus = time_step_vec*1e6

mask_stable_leapfrog = time_step_vec <= dt_CFL_traction
n_cases_stable_leapfrog = np.sum(mask_stable_leapfrog==True)

time_step_stable_leapfrog = time_step_vec[mask_stable_leapfrog]
