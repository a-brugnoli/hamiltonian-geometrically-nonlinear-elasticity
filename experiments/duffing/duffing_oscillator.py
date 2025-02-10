import numpy as np
from scipy.optimize import fsolve, root, approx_fprime
from scipy.special import ellipj
from scipy.integrate import simpson
from discrete_gradient import midpoint_discrete_gradient, mean_value_discrete_gradient

class DuffingOscillator:
    def __init__(self, alpha=1.0, beta=1.0, t_span=np.array([0, 1]), dt=0.01, q0=1):
        """
        Initialize Duffing oscillator parameters
        dq/dt = v
        dv/dt = - alpha*q - beta*q^3
        """
        self.alpha = alpha    # Linear stiffness
        self.beta = beta      # Nonlinear stiffness

        self.dt = dt
        simulation_time = t_span[1] - t_span[0]
        self.n_steps =np.round(simulation_time/dt).astype(int)
        T_init = t_span[0]
        T_end = self.n_steps*dt + T_init
        self.t_span = np.array([T_init, T_end])
        self.t_vec = np.linspace(T_init, T_end, self.n_steps+1)
        self.q0 = q0


    def jacobian_f(self, position):
        J = np.zeros((2, 2))
        J[0, 1] = 1
        J[1, 0] = -self.alpha - 3 * self.beta * position**2
        return J


    def hamiltonian(self, position, velocity):
        return 1/2 * velocity**2 + 0.5 * self.alpha * position**2 + 1/4 * self.beta * position**4


    def potential_energy(self, position):
        return 1/2 * self.alpha * position**2 + 1/4 * self.beta * position**4
        

    def grad_potential_energy(self, position):
        return self.alpha * position + self.beta * position**3


    def exact_solution(self):
        """
        Compute exact solution for unforced, undamped case
        using Jacobi elliptic functions
        """
        # Calculate modulus k for Jacobi elliptic functions
        if self.alpha > 0:
            k2 = self.beta * self.q0**2 / (2 * (self.alpha + self.beta * self.q0**2))
            omega0 = np.sqrt(self.alpha + self.beta * self.q0**2)
            
            # Use scipy's ellipj to compute Jacobi elliptic functions
            sn, cn, dn, _ = ellipj(omega0 * self.t_vec, k2)
            q_ex_vec = self.q0 * cn
            v_ex_vec = -omega0 * self.q0 * sn * dn
        else:
            raise ValueError("Damping must be positive")
            
        return q_ex_vec, v_ex_vec

   
    def leapfrog(self):
        """
        Solve using leapfrog/Verlet method
        Two version 
            - q at half-integers (the one introduced in the paper)
            - q at integers 
        Here we do at integers
        """
        q = np.zeros_like(self.t_vec)
        v = np.zeros_like(self.t_vec)
        q[0] = self.q0
        v[0] = 0
        
        # First half-step for velocity using Euler
        q_half = q[0] + 0.5 * self.dt * v[0]
        
        for i in range(self.n_steps):
            v[i+1] = v[i] - self.dt * self.grad_potential_energy(q_half)
            q_new_half = q_half + self.dt * v[i+1]
            q[i+1] = 0.5*(q_half + q_new_half)
            q_half = q_new_half

        return q, v


    def implicit_method(self, type="implicit midpoint"):
        """
        Solve using an implicit method. 
        - type : a string defining the method. Possible values are 
            - "implicit midpoint"
            - "midpoint discrete gradient"
            - "mean value discrete gradient"
    
        """
        q_vec = np.zeros_like(self.t_vec)
        v_vec = np.zeros_like(self.t_vec)
        q_vec[0] = self.q0
        v_vec[0] = 0

        def residual(y_new, y_old):
            q_old, v_old = y_old
            q_new, v_new = y_new

            q_mid = (q_old + q_new) / 2
            v_mid = (v_old + v_new) / 2

            if type == "implicit midpoint":
                dV_discrete = self.grad_potential_energy(q_mid)
            elif type == "midpoint discrete gradient":
                dV_discrete = midpoint_discrete_gradient(q_new, q_old, \
                                self.potential_energy, self.grad_potential_energy)
            elif type == "mean value discrete gradient":
                dV_discrete = mean_value_discrete_gradient(q_new, q_old, self.grad_potential_energy)
            else:
                raise ValueError("Unknown type of implicit method")
            
            res_q = q_new - q_old - self.dt * v_mid
            res_v = v_new - v_old + self.dt * dV_discrete
            return np.array([res_q, res_v])


        # def jacobian_implicit_midpoint(y_new, y_old):
        #     q_mid = (y_new[0] + y_old[0]) / 2   
        #     J = np.eye(2) - self.dt / 2 * self.jacobian_f(q_mid)
        #     return J
        
        # def numerical_jacobian(y, y_old, epsilon=1e-8):
        #     return approx_fprime(y, residual_implicit_midpoint, epsilon, y_old)

        # y_check = np.array([3,1])
        # y_old_check =np.array([2, 6])
        # J_analytic = jacobian_implicit_midpoint(y_check, y_old_check)
        # J_numerical = numerical_jacobian(y_check, y_old_check)
        # error_jacobian = np.linalg.norm(J_analytic - J_numerical) 
        # print(f"Error jacobian: {error_jacobian}")
        # assert error_jacobian < 1e-6

        perturbation = 1e-2
        for i in range(self.n_steps):
            x_old = np.array([q_vec[i], v_vec[i]])
            # Slightly perturbed previous guess to avoid division by zero
            # in the implicit midpoint method
            guess = np.array([q_vec[i]*(1 + perturbation), v_vec[i]])
            
            solution = fsolve(residual, guess, args=(x_old, ))
            q_vec[i+1], v_vec[i+1] = solution

            # Newton method
            # solution = root(residual_implicit_midpoint, guess, \
            #                 jac=jacobian_implicit_midpoint, args=(x_old, ))
            # q_array[i], v_array[i] = solution.x

        return q_vec, v_vec


    
    def energy_matrix(self):
        return np.diag([1, 1/self.alpha, 2/self.beta])


    def poisson_matrix(self, position):
        return np.array([[0, -1, -2*position], 
                         [1, 0, 0],
                         [2*position, 0, 0]])


    def linear_implicit(self):
        """
        Linear implicit method for Duffing oscillator
        
        Parameters
        ----------
        None
        
        Returns
        -------
        q, v : numpy arrays
            Position and velocity as a function of time
        """

        q_vec = np.zeros_like(self.t_vec)
        x_vec = np.zeros((len(self.t_vec), 3))

        q_vec[0] = self.q0

        sigma1_0 = self.alpha*self.q0
        sigma2_0 = 0.5*self.beta*self.q0**2
        x_vec[0, :] = np.array([0, sigma1_0, sigma2_0])

        # First half-step for velocity using Euler
        q_half = q_vec[0] + 0.5 * self.dt * x_vec[0, 0]

        H_matrix = self.energy_matrix()
        
        for i in range(self.n_steps):

            A = H_matrix - self.dt/2*self.poisson_matrix(q_half)
            B = H_matrix + self.dt/2*self.poisson_matrix(q_half)

            x_vec[i+1] = np.linalg.solve(A, B @ x_vec[i])

            q_new_half = q_half + self.dt * x_vec[i+1, 0]
            q_vec[i+1] = 0.5*(q_half + q_new_half)
            q_half = q_new_half

        return q_vec, x_vec


    def linear_implicit_static_condensation(self):
        """
        Linear implicit method for Duffing oscillator
        
        Parameters
        ----------
        None
        
        Returns
        -------
        q, v : numpy arrays
            Position and velocity as a function of time
        """

        q_vec = np.zeros_like(self.t_vec)
        v_vec = np.zeros(len(self.t_vec))
        sigma_vec = np.zeros((len(self.t_vec), 2))

        q_vec[0] = self.q0

        sigma1_0 = self.alpha*self.q0
        sigma2_0 = 0.5*self.beta*self.q0**2
        sigma_vec[0, :] = np.array([sigma1_0, sigma2_0])

        H_matrix = self.energy_matrix()
        M_compliance = H_matrix[1:, 1:]
        inv_M_C = np.diag(1 / np.diag(M_compliance))

        # First half-step for velocity using Euler
        q_half = q_vec[0] + 0.5 * self.dt * v_vec[0]
        
        for i in range(self.n_steps):

            L = np.array([1, 2*q_half])
            stiffness_matrix = L.T @ inv_M_C @ L
            

            A_vel = 1 + self.dt**2/4*stiffness_matrix
            B_vel = 1 - self.dt**2/4*stiffness_matrix

            v_vec[i+1] = (B_vel * v_vec[i] - self.dt*L.T @ sigma_vec[i])/A_vel
            sigma_vec[i+1] = sigma_vec[i] + self.dt * inv_M_C @ L * (v_vec[i+1] + v_vec[i])/2

            q_new_half = q_half + self.dt * v_vec[i+1]
            q_vec[i+1] = 0.5*(q_half + q_new_half)
            q_half = q_new_half

        x_vec = np.column_stack((v_vec, sigma_vec))

        return q_vec, x_vec
