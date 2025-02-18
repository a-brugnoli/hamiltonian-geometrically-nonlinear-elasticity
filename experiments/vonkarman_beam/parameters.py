import numpy as np
from math import pi


width = 0.002 # m
height = 0.002 # m
A = width*height # Cross section
I = (width * height**3) / 12 # Second moment of inertia
rho = 2700 # Density [kg/m^3]
E = 70*10**9 # Young modulus [Pa]

wave_speed_bending = np.sqrt(E*I/(rho*A))
wave_speed_traction = np.sqrt(E/rho)

L = 1
n_elements = 50

n_dofs_hor = n_elements+1
n_dofs_ver = 2*(n_elements+1)
mesh_size = L/n_elements
# # In finite elements, the minimum security coefficient is:
# - 4 for bending 
# - 2 for traction
sec_coeff_bend = 4
sec_coeff_traction = 2


dt_max_bending = mesh_size**2/(2*wave_speed_bending)
dt_max_traction = mesh_size/wave_speed_traction

dt_CFL_bending = dt_max_bending/sec_coeff_bend
dt_CFL_traction = dt_max_traction/sec_coeff_traction

# # Initial condition
alpha = 10
ampl_hor_disp_0 = 0 # alpha*height
ampl_ver_disp_0 = alpha*height

# # Time step to have sufficient resolution
dt_base = dt_CFL_bending

omega1_bending = (pi/L)**2*wave_speed_bending
T1_bending = 2*pi/omega1_bending



t_end_approx = T1_bending
n_steps_approx = np.round(t_end_approx/dt_base).astype(int)

max_output = 1e4
if n_steps_approx+1<=max_output:
    n_steps = n_steps_approx
    n_sim_output = n_steps+1
else:
    n_sim_output = max_output
    n_steps_output = n_sim_output-1
    n_steps = np.round(n_steps_approx/n_steps_output)*n_steps_output

t_end = n_steps.astype(int)*dt_base
t_span = [0, t_end]
