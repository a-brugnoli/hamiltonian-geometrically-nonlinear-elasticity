import numpy as np
import os
from math import pi 

# Initial condition
q0 = 10

# Pyisical parameters
alpha = 10
beta = 5
omega_0 = np.sqrt(alpha + beta * q0**2)

T = 2*pi/omega_0
t_end = T*100
# Time parameters
t_span = [0, t_end]

norm_type = "L2" 
dt_base = T/100
# sec_factor = 1/10
# dt_base = sec_factor*2/omega_0
n_case = 5
log_base = 2
time_step_vec = [dt_base/log_base**n for n in range(n_case)]

directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/"
if not os.path.exists(directory_results):
    os.makedirs(directory_results)

file_time = directory_results + "results_time.pkl"
file_results_position = directory_results + "results_position.pkl"
file_results_velocity = directory_results + "results_velocity.pkl"
file_results_energy = directory_results + "results_energy.pkl"

file_results_error_position = directory_results + "results_error_position.pkl"
file_results_error_velocity = directory_results + "results_error_velocity.pkl"
file_results_error_energy = directory_results + "results_error_energy.pkl"
file_results_comp_time = directory_results + "results_comp_time.pkl"


directory_images = f"{os.path.dirname(os.path.abspath(__file__))}/images/"
if not os.path.exists(directory_images):
    os.makedirs(directory_images)