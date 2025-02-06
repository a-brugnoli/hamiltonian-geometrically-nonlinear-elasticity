import numpy as np
from scipy.optimize import fsolve, root, approx_fprime
from scipy.special import ellipj
from discrete_gradient import midpoint_discrete_gradient

class DuffingOscillator:
    def __init__(self, alpha=1.0, beta=1.0, t_span=1, dt=0.01, q0=1):
        """
        Initialize Duffing oscillator parameters
        dq/dt = v
        dv/dt = - alpha*q - beta*q^3
        """
        self.alpha = alpha    # Linear stiffness
        self.beta = beta      # Nonlinear stiffness
        self.t_span = t_span
        self.dt = dt
        self.n_steps = int((t_span[1] - t_span[0])/dt)
        self.t_vec = np.linspace(t_span[0], t_span[1], self.n_steps)
        self.q0 = q0


    def rhs(self, position):
        return - self.alpha * position - self.beta * position**3


    def jacobian_f(self, position):

        J = np.zeros((2, 2))
        J[0, 1] = 1
        J[1, 0] = -self.alpha - 3 * self.beta * position**2
        return J

    def hamiltonian(self, position, velocity):
        return 1/2 * velocity**2 + 0.5 * self.alpha * position**2 + 1/4 * self.beta * position**4


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
        """
        q = np.zeros(self.n_steps)
        v = np.zeros(self.n_steps)
        q[0] = self.q0
        v[0] = 0
        
        # First half-step for velocity using Euler
        q_half = q[0] + 0.5 * self.dt * v[0]
        # v_half = v[0] + 0.5 * dt * self.rhs(q[0])
        
        for i in range(1, self.n_steps):
            v[i] = v[i-1] + self.dt * self.rhs(q_half)
            q_new_half = q_half + self.dt * v[i]
            q[i] = 0.5*(q_half + q_new_half)
            q_half = q_new_half

            # # Version where q iss computed at integers
            # # Position update
            # q[i] = q[i-1] + dt * v_half
            # # Velocity update
            # v_new_half = v_half + dt * self.rhs(q[i])
            # v[i] = (v_half + v_new_half) / 2
            # v_half = v_new_half
            
        return q, v


    def implicit_midpoint(self):
        """
        Solve using implicit midpoint method
        """
        q = np.zeros(self.n_steps)
        v = np.zeros(self.n_steps)
        q[0] = self.q0
        v[0] = 0

        def residual_implicit_midpoint(y, y_old):
            q_mid = (y[0] + y_old[0]) / 2
            v_mid = (y[1] + y_old[1]) / 2
            
            res_q = y[0] - y_old[0] - self.dt * v_mid
            res_v = y[1] - y_old[1] - self.dt * self.rhs(q_mid)
            return np.array([res_q, res_v])


        def jacobian_implicit_midpoint(y, y_old):
            q_mid = (y[0] + y_old[0]) / 2   
            J = np.eye(2) - self.dt / 2 * self.jacobian_f(q_mid)
            return J
        

        # def numerical_jacobian(y, y_old, epsilon=1e-8):
        #     return approx_fprime(y, residual_implicit_midpoint, epsilon, y_old)

        # y_check = np.array([3,1])
        # y_old_check =np.array([2, 6])
        # J_analytic = jacobian_implicit_midpoint(y_check, y_old_check)
        # J_numerical = numerical_jacobian(y_check, y_old_check)
        # error_jacobian = np.linalg.norm(J_analytic - J_numerical) 
        # print(f"Error jacobian: {error_jacobian}")
        # assert error_jacobian < 1e-6

        for i in range(1, self.n_steps):
            # Use previous values as initial guess
            guess = np.array([q[i-1], v[i-1]])
            # solution = root(residual_implicit_midpoint, guess, \
            #                 jac=jacobian_implicit_midpoint, args=(guess, ))
            # q[i], v[i] = solution.x
            
            solution = fsolve(residual_implicit_midpoint, guess, args=(guess, ))
            q[i], v[i] = solution

        return q, v


    def midpoint_discrete_gradient(self):
        """
        Solve using midpoint discrete gradient method
        """
        q = np.zeros(self.n_steps)
        v = np.zeros(self.n_steps)
        
        q[0] = self.q0
        v[0] = 0

        # def discrete_potential_gradient(q_new, q_old):
        #     """
        #     Midpoint discrete gradient of potential energy V(q) = (α/2)q² + (β/4)q⁴
        #     Using averaged form that satisfies the discrete gradient property
        #     """
        #     # For quadratic term (α/2)x²
        #     grad_quad = self.alpha * (q_old + q_new) / 2
            
        #     # For quartic term (β/4)x⁴
        #     grad_quart = self.beta * (q_old**3 + q_old**2*q_new + q_old*q_new**2 + q_new**3) / 4

        #     return grad_quad + grad_quart

        
        
        def residual_midpoint_discrete_gradient(y_new, y_old):
            q_old, v_old = y_old
            q_new, v_new = y_new

            dV_discrete = midpoint_discrete_gradient(q_new, q_old)
            # dV_discrete = discrete_potential_gradient(q_new, q_old)
            
            res_q = q_new - q_old - self.dt * (v_new + v_old) / 2
            res_v = v_new - v_old + self.dt * dV_discrete
            return np.array([res_q, res_v])
        

        for i in range(1, self.n_steps):
            guess = np.array([q[i-1], v[i-1]])
            solution = fsolve(residual_midpoint_discrete_gradient, guess, args=(guess, ))
            q[i], v[i] = solution

            print(f"q: {q[i]}, v: {v[i]}")
            
        return q, v


    # def mean_value_discrete_gradient(self):
    #     """
    #     Solve using mean value discrete gradient method
    #     """
    #     n_steps = int((self.t_span[1] - self.t_span[0]) / dt)
    #     t = np.linspace(self.t_span[0], self.t_span[1], n_steps)
    #     q = np.zeros(n_steps)
    #     v = np.zeros(n_steps)
        
    #     q[0] = q0
    #     v[0] = v0

    #     def mean_value_gradient(x1, x2):
    #         """
    #         Mean value discrete gradient of potential energy V(x) = (α/2)x² + (β/4)x⁴
    #         ∇V(x₁,x₂) = ∫_0^1  ∇V(sx₁ + (1-s)x₂)ds
    #         """
    #         # For quadratic term (α/2)x²
    #         grad_quad = self.alpha * (x1 + x2) / 2
            
    #         # For quartic term (β/4)x⁴, compute integral numerically
    #         n_points = 100
    #         s = np.linspace(0, 1, n_points)
    #         xs = x2 + s * (x1 - x2)  # Linear interpolation between x2 and x1
    #         grad_quart = self.beta * np.mean(xs**3)
            
    #         return grad_quad + grad_quart

    #     def residual(y):
    #         q_new, v_new = y
    #         dV = mean_value_gradient(q_new, q[i-1])
            
    #         rq = q_new - q[i-1] - dt * (v_new + v[i-1]) / 2
    #         rv = v_new - v[i-1] + dt * dV
    #         return [rq, rv]

    #     for i in range(1, n_steps):
    #         guess = [q[i-1], v[i-1]]
    #         solution = fsolve(residual, guess)
    #         q[i], v[i] = solution
            
    #     return t, q, v

    