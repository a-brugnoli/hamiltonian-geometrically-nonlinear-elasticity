import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj
import time

def exact_solution(q0, alpha, beta, t):
    """
    Computes the exact solution of the nonlinear Duffing oscillator using Jacobi elliptic functions.
    Parameters:
    q0 (float): Initial displacement.
    alpha (float): Linear stiffness coefficient.
    beta (float): Nonlinear stiffness coefficient.
    t (float or np.ndarray): Time point(s) at which to evaluate the solution.
    Returns:
    q_exact (np.ndarray or float): Displacement over time.
    v_exact (np.ndarray or float): Velocity over time.
    """
    omega_0 = np.sqrt(alpha + beta * q0**2)
    k_sq = beta * q0**2 / (2 * (alpha + beta * q0**2))
    sn, cn, dn, _ = ellipj(omega_0 * t, k_sq)
    q_exact = q0 * cn
    v_exact = -omega_0 * q0 * sn * dn
    return q_exact, v_exact


def simulate_proposed(alpha, beta, q0, dt, n_samples):
    """
    Simulates the Duffing oscillator using a structure-preserving scheme.
    """
    qh, _ = exact_solution(q0, alpha, beta, 0.5 * dt)
    xn = np.zeros((3, 1))
    xn[1] = 0.5 * alpha * q0
    xn[2] = 0.5 * beta * q0**2
    H = np.diag([1, 4 / alpha, 2 / beta])
    J0 = np.zeros((3, 3))
    J0[1, 0] = 2.0
    J0[0, 1] = -2.0
    A0 = H - 0.5 * dt * J0
    B0 = H + 0.5 * dt * J0
    displacement = np.zeros((n_samples, 1))
    velocity = np.zeros((n_samples, 1))
    for n in range(n_samples):
        dtqh = dt * qh
        A = A0.copy()
        B = B0.copy()
        A[0, 2], A[2, 0] = dtqh, -dtqh
        B[0, 2], B[2, 0] = -dtqh, dtqh
        xnp1 = np.linalg.solve(A, B @ xn)
        v_n = float(xnp1[0, 0])
        qh_next = qh + dt * v_n
        displacement[n] = 0.5 * (qh + qh_next)
        velocity[n] = v_n
        qh = qh_next
        xn = xnp1
    return displacement, velocity


def simulate_proposed_static_condensation(alpha, beta, q0, dt, n_samples):
    """
    Simulates the Duffing oscillator using a structure-preserving scheme.
    """
    qh, _ = exact_solution(q0, alpha, beta, 0.5 * dt)
    v_n = 0
    sigma_n = np.array([alpha * q0, 0.5 * beta * q0**2])

    inv_M_C = np.diag([alpha, beta/2])
    
    L_mat = np.array([dt, 0])

    displacement = np.zeros((n_samples, 1))
    velocity = np.zeros((n_samples, 1))

    A0_vel = 1 + dt**2/4*alpha
    B0_vel = 1 - dt**2/4*alpha

    for n in range(n_samples):
        dtqh = dt * qh
        A_vel = A0_vel + dtqh**2 * inv_M_C[1, 1]
        B_vel = B0_vel - dtqh**2 * inv_M_C[1, 1]
        L_mat[1] = 2 * dtqh

        v_np1 = (B_vel * v_n - L_mat.T @ sigma_n)/A_vel
        sigma_np1 = sigma_n + inv_M_C @ L_mat * (v_np1 + v_n)/2

        v_n = v_np1
        qh_next = qh + dt * v_n
        displacement[n] = 0.5 * (qh + qh_next)
        velocity[n] = v_n
        qh = qh_next
        sigma_n = sigma_np1

    return displacement, velocity



def simulate_semiexplicit(alpha, beta, q0, dt, n_samples):
    """
    Simulates the Duffing oscillator using a semi-explicit integration scheme.
    """
    sqrtalpha = np.sqrt(alpha)
    sqrttwobeta = np.sqrt(2 * beta)
    q0, _ = exact_solution(q0, alpha, beta, 0)
    qh, _ = exact_solution(q0, alpha, beta, 0.5 * dt)
    velocity = np.zeros((n_samples, 1))
    displacement = np.zeros((n_samples, 1))
    vprev = 0
    psilprev = sqrtalpha * qh
    psinlprev = np.sqrt(0.5 * beta) * q0**2
    for n in range(n_samples):
        g = sqrttwobeta * qh
        numv = (1 - 0.25 * dt**2 * g**2) * vprev - dt * sqrtalpha * psilprev - dt * g * psinlprev
        denv = 1 + 0.25 * dt**2 * g**2
        vnext = numv / denv
        psilnext = psilprev + sqrtalpha * dt * vnext
        psinlnext = psinlprev + 0.5 * g * dt * (vnext + vprev)
        qhnext = qh + dt * vnext
        displacement[n] = 0.5 * (qh + qhnext)
        velocity[n] = vnext
        vprev = vnext
        qh = qhnext
        psilprev = psilnext
        psinlprev = psinlnext


    return displacement, velocity


def run_convergence_tests(alpha, beta, q0, T_end, N_trials):
    """
    Runs convergence and timing tests over a range of time steps.
    """
    dt_vec = np.logspace(-2, -5, N_trials)
    errMat = np.zeros((N_trials, 4))
    timeMat = np.zeros((N_trials, 2))
    for i, dt in enumerate(dt_vec):
        n_samples = int(np.round(T_end / dt))
        T_adj = n_samples * dt
        print(f"[{i+1}/{N_trials}] dt = {dt:.5e}, T_end = {T_adj:.5f}")
        t0 = time.perf_counter()
        # disp_p, vel_p = simulate_proposed(alpha, beta, q0, dt, n_samples)
        disp_p, vel_p = simulate_proposed_static_condensation(alpha, beta, q0, dt, n_samples)
        t1 = time.perf_counter()
        timeMat[i, 0] = (t1 - t0) * 1000
        t0 = time.perf_counter()
        disp_sav, vel_sav = simulate_semiexplicit(alpha, beta, q0, dt, n_samples)
        t1 = time.perf_counter()
        timeMat[i, 1] = (t1 - t0) * 1000
        q_ex, v_ex = exact_solution(q0, alpha, beta, T_adj)
        errMat[i, 0] = np.abs(disp_p[-1, 0] - q_ex)
        errMat[i, 1] = np.abs(vel_p[-1, 0] - v_ex)
        errMat[i, 2] = np.abs(disp_sav[-1, 0] - q_ex)
        errMat[i, 3] = np.abs(vel_sav[-1, 0] - v_ex)

    return dt_vec, errMat, timeMat


def main():
    alpha = 5
    beta = 10
    q0 = 10

    omega_0 = np.sqrt(alpha + beta * q0**2)
    T = 2 * np.pi / omega_0
    T_end = 100 * T
    dt = T / 100
    n_samples = int(np.round(T_end / dt))
    time_vector = np.linspace(0.0, T_end, n_samples)

    # disp_p, vel_p = simulate_proposed(alpha, beta, q0, dt, n_samples)
    disp_p, vel_p = simulate_proposed_static_condensation(alpha, beta, q0, dt, n_samples)

    disp_sav, vel_sav = simulate_semiexplicit(alpha, beta, q0, dt, n_samples)
    q_ex, v_ex = exact_solution(q0, alpha, beta, time_vector + 0.5 * dt)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(time_vector, disp_p, label='Proposed', linewidth=0.8)
    axs[0].plot(time_vector, disp_sav, '--', label='Semi-explicit', linewidth=0.8)
    axs[0].plot(time_vector, q_ex, '--', label='Exact', linewidth=0.8)
    axs[0].set_ylabel("Displacement")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_vector, vel_p, label='Proposed', linewidth=0.8)
    axs[1].plot(time_vector, vel_sav, '--', label='Semi-explicit', linewidth=0.8)
    axs[1].plot(time_vector, v_ex, '--', label='Exact', linewidth=0.8)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("Duffing Oscillator: Numerical vs Exact")
    plt.tight_layout()
    plt.xlim([0, 0.75])
    plt.show()

    N_trials = 20
    dt_vec, errMat, timeMat = run_convergence_tests(alpha, beta, q0, T_end, N_trials)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].loglog(dt_vec, errMat[:, 0], '-', label='Proposed', linewidth=0.8)
    axs[0].loglog(dt_vec, errMat[:, 2], '--', label='Semi-explicit', linewidth=0.8)

    axs[0].set_ylabel("Displacement Error")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].loglog(dt_vec, errMat[:, 1], '-', label='Proposed', linewidth=0.8)
    axs[1].loglog(dt_vec, errMat[:, 3], '--', label='Semi-explicit', linewidth=0.8)
    axs[1].set_ylabel("Velocity Error")
    axs[1].set_xlabel("Time Step Size (dt)")
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("Convergence: Error vs Time Step")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.loglog(dt_vec, timeMat[:, 0], '-', label='Proposed', linewidth=0.8)
    plt.loglog(dt_vec, timeMat[:, 1], '-', label='Semi-explicit', linewidth=0.8)
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("Computation Time (ms)")
    plt.title("Computation Time vs Time Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()