import numpy as np
import time
import matplotlib.pyplot as plt
from src.postprocessing.options import configure_matplotib
configure_matplotib()
from duffing_oscillator import DuffingOscillator

# Time parameters
t_span = [0, 5]
dt = 0.1
# Pyisical parameters
alpha = 1
beta = 1
# Initial condition
q0 = 1

duffing = DuffingOscillator(alpha, beta, t_span, dt, q0)
t_vec = duffing.t_vec


# Compute exact solution and numerical solutions
q_exact, v_exact = duffing.exact_solution()

t0_leapfrog = time.perf_counter()
q_leapfrog, v_leapfrog = duffing.leapfrog()
tf_leapfrog = time.perf_counter()
elapsed_leapfrog = tf_leapfrog - t0_leapfrog

t0_imp_midpoint = time.perf_counter()
q_imp_midpoint, v_imp_midpoint = duffing.implicit_midpoint()
tf_imp_midpoint = time.perf_counter()
elapsed_imp_midpoint = tf_imp_midpoint - t0_imp_midpoint

t0_mid_dg = time.perf_counter()
q_mid_dg, v_mid_dg = duffing.midpoint_discrete_gradient()
tf_mid_dg = time.perf_counter()
elapsed_mid_dg = tf_mid_dg - t0_mid_dg

print(f"Elapsed time Leapfrog: {elapsed_leapfrog}")
print(f"Elapsed time Implicit Midpoint: {elapsed_imp_midpoint}")
print(f"Elapsed time Midpoint Discrete gradient: {elapsed_imp_midpoint}")

# t_mean_dg, q_mean_dg, v_mean_dg = duffing.mean_value_discrete_gradient()

# Compute energies
E_exact = duffing.hamiltonian(q_exact, v_exact)
E_leapfrog = duffing.hamiltonian(q_leapfrog, v_leapfrog)
E_imp_midpoint = duffing.hamiltonian(q_imp_midpoint, v_imp_midpoint)
E_mid_dg = duffing.hamiltonian(q_mid_dg, v_mid_dg)
# E_mean_dg = duffing.hamiltonian(q_mean_dg, v_mean_dg)

# Create plots
plt.figure(figsize=(16, 8))
# Position plot
plt.subplot(2, 2, 1)
plt.plot(t_vec, q_exact, 'k--', label='Exact', linewidth=2)
# plt.plot(t_vec, q_leapfrog, label='Leapfrog')
# plt.plot(t_vec, q_imp_midpoint, label='Implicit Midpoint')
plt.plot(t_vec, q_mid_dg, label='Midpoint DG')
# plt.plot(t_mean_dg, q_mean_dg, label='Mean Value DG')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.title('Position vs Time')

# Position error
plt.subplot(2, 2, 2)
# plt.semilogy(t_vec, np.abs(q_leapfrog - q_exact), label='Leapfrog')
# plt.semilogy(t_vec, np.abs(q_imp_midpoint - q_exact), label='Implicit Midpoint')
plt.semilogy(t_vec, np.abs(q_mid_dg - q_exact), label='Midpoint DG')
# plt.plot(t_mean_dg, np.abs(q_mean_dg - q_exact), label='Mean Value DG')
plt.xlabel('Time')
plt.ylabel('Position Error')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.title('Position Error vs Time')

# Velocity plot
plt.subplot(2, 2, 3)
plt.plot(t_vec, v_exact, 'k--', label='Exact', linewidth=2)
# plt.plot(t_vec, v_leapfrog, label='Leapfrog')
# plt.plot(t_vec, v_imp_midpoint, label='Implicit Midpoint')
plt.plot(t_vec, v_mid_dg, label='Midpoint DG')
# plt.plot(t_mean_dg, v_mean_dg, label='Mean Value DG')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.title('Velocity vs Time')

# Velocity error
plt.subplot(2, 2, 4)
# plt.semilogy(t_vec, np.abs(v_leapfrog - v_exact), label='Leapfrog')
# plt.semilogy(t_vec, np.abs(v_imp_midpoint - v_exact), label='Implicit Midpoint')
plt.semilogy(t_vec, np.abs(v_mid_dg - v_exact), label='Midpoint DG')
# plt.plot(t_mean_dg, np.abs(v_mean_dg - v_exact), label='Mean Value DG')
plt.xlabel('Time')
plt.ylabel('Velocity Error')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.title('Velocity Error vs Time')


plt.figure(figsize=(16, 8))
# Phase space plot
plt.subplot(1, 2, 1)
plt.plot(q_exact, v_exact, 'k--', label='Exact', linewidth=2)
# plt.plot(q_leapfrog, v_leapfrog, label='Leapfrog')
# plt.plot(q_imp_midpoint, v_imp_midpoint, label='Implicit Midpoint')
plt.plot(q_mid_dg, v_mid_dg, label='Midpoint DG')
# plt.plot(q_mean_dg, v_mean_dg, label='Mean Value DG')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.title('Phase Space')

# Energy error
plt.subplot(1, 2, 2)
# plt.semilogy(t_vec, np.abs(E_leapfrog - E_exact[0])/E_exact[0], label='Leapfrog')
# plt.semilogy(t_vec, np.abs(E_imp_midpoint - E_exact[0])/E_exact[0], label='Implicit Midpoint')
plt.semilogy(t_vec, np.abs(E_mid_dg - E_exact[0])/E_exact[0], label='Midpoint DG')
# plt.plot(t_mean_dg, np.abs(E_mean_dg - E_exact[0])/E_exact[0], label='Mean Value DG')
plt.xlabel('Time')
plt.ylabel('Relative Energy Error')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.title('Energy Error vs Time')

plt.show()
