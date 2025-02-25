import matplotlib.pyplot as plt
import numpy as np
from src.postprocessing.options import configure_matplotib
import pickle
configure_matplotib()
from experiments.duffing.parameters_duffing import *

with open(file_results_energy, "rb") as f:
        dict_energy = pickle.load(f)

with open(file_results_position, "rb") as f:
        dict_position = pickle.load(f)

with open(file_results_velocity, "rb") as f:
        dict_velocity = pickle.load(f)

with open(file_time, "rb") as f:
        dict_time = pickle.load(f)

# Create plots
t_vec = dict_time["Time"]

q_exact = dict_position["Exact"]
q_leapfrog = dict_position["Leapfrog"]
q_dis_gradient = dict_position["Discrete gradient"]
q_lin_implicit = dict_position["Linear implicit"]

v_exact = dict_velocity["Exact"]
v_leapfrog = dict_velocity["Leapfrog"]
v_dis_gradient = dict_velocity["Discrete gradient"]
v_lin_implicit = dict_velocity["Linear implicit"]

E_exact = dict_energy["Exact"]
E_leapfrog = dict_energy["Leapfrog"]
E_dis_gradient = dict_energy["Discrete gradient"]
E_lin_implicit = dict_energy["Linear implicit"]

plt.figure(figsize=(16, 8))
# Position plot
plt.subplot(2, 2, 1)
plt.plot(t_vec, q_exact, 'k--', label='Exact', linewidth=2)
plt.plot(t_vec, q_dis_gradient, label='Midpoint DG')
plt.plot(t_vec, q_lin_implicit, label='Lin implicit')
plt.plot(t_vec, q_leapfrog, label='Leapfrog')


plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.title('Position vs Time')

# Position error
plt.subplot(2, 2, 2)
plt.semilogy(t_vec, np.abs(q_dis_gradient - q_exact), label='Midpoint DG')
plt.semilogy(t_vec, np.abs(q_lin_implicit - q_exact), label='Lin implicit')
plt.semilogy(t_vec, np.abs(q_leapfrog - q_exact), label='Leapfrog')

plt.xlabel('Time')
plt.ylabel('Position Error')
plt.legend()
plt.grid(True)
plt.title('Position Error vs Time')

# Velocity plot
plt.subplot(2, 2, 3)
plt.plot(t_vec, v_exact, 'k--', label='Exact', linewidth=2)
plt.plot(t_vec, v_dis_gradient, label='Midpoint DG')
plt.plot(t_vec, v_lin_implicit, label='Lin implicit')
plt.plot(t_vec, v_leapfrog, label='Leapfrog')

plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.title('Velocity vs Time')

# Velocity error
plt.subplot(2, 2, 4)
plt.semilogy(t_vec, np.abs(v_dis_gradient - v_exact), label='Midpoint DG')
plt.semilogy(t_vec, np.abs(v_lin_implicit - v_exact), label='Lin implicit')
plt.semilogy(t_vec, np.abs(v_leapfrog - v_exact), label='Leapfrog')

plt.xlabel('Time')
plt.ylabel('Velocity Error')
plt.legend()
plt.grid(True)
plt.title('Velocity Error vs Time')


plt.figure(figsize=(16, 8))
# Phase space plot
plt.subplot(1, 2, 1)
plt.plot(q_exact, v_exact, 'k--', label='Exact', linewidth=2)
plt.plot(q_dis_gradient, v_dis_gradient, label='Midpoint DG')
plt.plot(q_lin_implicit, v_lin_implicit, label='Lin implicit')
plt.plot(q_leapfrog, v_leapfrog, label='Leapfrog')

plt.xlabel('Position')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.title('Phase Space')


# Energy error
plt.subplot(1, 2, 2)
plt.semilogy(t_vec, np.abs(E_dis_gradient - E_exact)/E_exact, label='Midpoint DG')
plt.semilogy(t_vec, np.abs(E_lin_implicit - E_exact)/E_exact, label='Lin implicit')
plt.semilogy(t_vec, np.abs(E_leapfrog - E_exact)/E_exact, label='Leapfrog')

plt.xlabel('Time')
plt.ylabel('Relative Energy Error')
plt.legend()
plt.grid(True)
plt.title('Energy Error vs Time')

plt.show()
