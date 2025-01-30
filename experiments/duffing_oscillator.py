import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import ellipj
from src.postprocessing.options import configure_matplotib
configure_matplotib()


class DuffingOscillator:
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Initialize Duffing oscillator parameters
        dx/dt = v
        dv/dt = - alpha*q - beta*x^3
        """
        self.alpha = alpha    # Linear stiffness
        self.beta = beta      # Nonlinear stiffness

    
    def rhs(self, position):
        return - self.alpha * position - self.beta * position**3


    def energy(self, position, velocity):
        return 1/2 * velocity**2 + 0.5 * self.alpha * position**2 + 1/4 * self.beta * position**4
    

    # def exact_solution(self, q0, v0, t):
    #     """
    #     Compute exact solution for unforced, undamped case
    #     using Jacobi elliptic functions
    #     """
    #     # Calculate energy from initial conditions
    #     assert v0==0
    #     # Calculate modulus k for Jacobi elliptic functions
    #     if self.alpha > 0:
    #         k2 = self.beta * q0**2 / (2 * (self.alpha + self.beta * q0**2))
    #         omega0 = np.sqrt(self.alpha + self.beta * q0**2)
            
    #         # Use scipy's ellipj to compute Jacobi elliptic functions
    #         sn, cn, dn, ph = ellipj(omega0 * t, k2)
    #         q = q0 * cn
    #         v = -omega0 * q0 * sn * dn
    #     else:
    #         raise ValueError("Damping must be positive")
            
    #     return q, v


    def exact_solution(self, x0, v0, t):
        """
        Compute the analytical solution at specified time points.
        
        Parameters:
        -----------
        t : array-like
            Time points
        x0 : float
            Initial position
        v0 : float
            Initial velocity
            
        Returns:
        --------
        tuple (x, v)
            Position and velocity at requested time points
        """
        # Calculate the total energy (constant of motion)
        E = 0.5 * v0**2 + 0.5 * self.alpha * x0**2 + 0.25 * self.beta * x0**4
        
        # Calculate the modulus k for Jacobi elliptic functions
        if x0 == 0:
            # Special case: pure oscillation from velocity
            x_max = np.sqrt(-2*self.alpha/self.beta + 
                          np.sqrt(4*self.alpha**2/self.beta**2 + 8*E/self.beta))
        else:
            x_max = x0
        
        k = np.sqrt((x_max**2 * self.beta)/(2 * (self.alpha + self.beta * x_max**2)))
        
        # Calculate frequency
        omega = np.sqrt(self.alpha + self.beta * x_max**2)
        
        # Phase shift to match initial conditions
        if x0 == 0:
            phi = np.pi/2 if v0 > 0 else -np.pi/2
        else:
            phi = 0 if v0 == 0 else np.arctan2(v0, omega * x0)
        
        # Calculate position using Jacobi elliptic functions
        sn, cn, dn, _ = ellipj(omega * t + phi, k**2)
        x = x_max * cn
        
        # Calculate velocity
        v = -omega * x_max * sn * dn
        
        return x, v



    def leapfrog(self, q0, v0, t_span, dt):
        """
        Solve using leapfrog/Verlet method
        """
        n_steps = int((t_span[1] - t_span[0]) / dt)
        t = np.linspace(t_span[0], t_span[1], n_steps)
        q = np.zeros(n_steps)
        v = np.zeros(n_steps)
        
        q[0] = q0
        v[0] = v0
        
        # First half-step for velocity using Euler
        v_half = v[0] + 0.5 * dt * self.rhs(q[0])
        
        for i in range(1, n_steps):
            # Position update
            q[i] = q[i-1] + dt * v_half
            
            # Velocity update
            v_new_half = v_half + dt * self.rhs(q[i])
            
            v[i] = (v_half + v_new_half) / 2
            v_half = v_new_half
            
        return t, q, v

    def implicit_midpoint(self, q0, v0, t_span, dt):
        """
        Solve using implicit midpoint method
        """
        n_steps = int((t_span[1] - t_span[0]) / dt)
        t = np.linspace(t_span[0], t_span[1], n_steps)
        q = np.zeros(n_steps)
        v = np.zeros(n_steps)
        
        q[0] = q0
        v[0] = v0

        def residual(y):
            q_mid = (y[0] + q[i-1]) / 2
            v_mid = (y[1] + v[i-1]) / 2
            
            rq = y[0] - q[i-1] - dt * v_mid
            rv = y[1] - v[i-1] - dt * self.rhs(q_mid)
            return [rq, rv]

        for i in range(1, n_steps):
            # Use previous values as initial guess
            guess = [q[i-1], v[i-1]]
            solution = fsolve(residual, guess)
            q[i], v[i] = solution
            
        return t, q, v

    def midpoint_discrete_gradient(self, q0, v0, t_span, dt):
        """
        Solve using midpoint discrete gradient method
        """
        n_steps = int((t_span[1] - t_span[0]) / dt)
        t = np.linspace(t_span[0], t_span[1], n_steps)
        q = np.zeros(n_steps)
        v = np.zeros(n_steps)
        
        q[0] = q0
        v[0] = v0

        def discrete_potential_gradient(x1, x2):
            """
            Midpoint discrete gradient of potential energy V(x) = (α/2)x² + (β/4)x⁴
            Using averaged form that satisfies the discrete gradient property
            """
            # For quadratic term (α/2)x²
            grad_quad = self.alpha * (x1 + x2) / 2
            
            # For quartic term (β/4)x⁴
            grad_quart = self.beta * (x1**3 + x1**2*x2 + x1*x2**2 + x2**3) / 4
            
            return grad_quad + grad_quart

        def residual(y):
            q_new, v_new = y
            dV = discrete_potential_gradient(q_new, q[i-1])
            
            rq = q_new - q[i-1] - dt * (v_new + v[i-1]) / 2
            rv = v_new - v[i-1] + dt * dV
            return [rq, rv]

        for i in range(1, n_steps):
            guess = [q[i-1], v[i-1]]
            solution = fsolve(residual, guess)
            q[i], v[i] = solution
            
        return t, q, v

    def mean_value_discrete_gradient(self, q0, v0, t_span, dt):
        """
        Solve using mean value discrete gradient method
        """
        n_steps = int((t_span[1] - t_span[0]) / dt)
        t = np.linspace(t_span[0], t_span[1], n_steps)
        q = np.zeros(n_steps)
        v = np.zeros(n_steps)
        
        q[0] = q0
        v[0] = v0

        def mean_value_gradient(x1, x2):
            """
            Mean value discrete gradient of potential energy V(x) = (α/2)x² + (β/4)x⁴
            ∇V(x₁,x₂) = ∫₀¹ ∇V(sx₁ + (1-s)x₂)ds
            """
            # For quadratic term (α/2)x²
            grad_quad = self.alpha * (x1 + x2) / 2
            
            # For quartic term (β/4)x⁴, compute integral numerically
            n_points = 100
            s = np.linspace(0, 1, n_points)
            xs = x2 + s * (x1 - x2)  # Linear interpolation between x2 and x1
            grad_quart = self.beta * np.mean(xs**3)
            
            return grad_quad + grad_quart

        def residual(y):
            q_new, v_new = y
            dV = mean_value_gradient(q_new, q[i-1])
            
            rq = q_new - q[i-1] - dt * (v_new + v[i-1]) / 2
            rv = v_new - v[i-1] + dt * dV
            return [rq, rv]

        for i in range(1, n_steps):
            guess = [q[i-1], v[i-1]]
            solution = fsolve(residual, guess)
            q[i], v[i] = solution
            
        return t, q, v

def compare_all_methods():
    # Initialize oscillator with no damping or forcing
    duffing = DuffingOscillator(alpha=1.0, beta=1.0)
    
    # Time parameters
    t_span = [0, 5]
    dt = 0.01
    t = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0])/dt))
    
    # Initial conditions
    q0, v0 = 1.0, 0.1

    # assert v0==0
    
    # Compute exact solution and numerical solutions
    q_exact, v_exact = duffing.exact_solution(q0, v0, t)
    t_leapfrog, q_leapfrog, v_leapfrog = duffing.leapfrog(q0, v0, t_span, dt)
    t_imp_midpoint, q_imp_midpoint, v_imp_midpoint = duffing.implicit_midpoint(q0, v0, t_span, dt)
    t_mid_dg, q_mid_dg, v_mid_dg = duffing.midpoint_discrete_gradient(q0, v0, t_span, dt)
    t_mean_dg, q_mean_dg, v_mean_dg = duffing.mean_value_discrete_gradient(q0, v0, t_span, dt)
    
    # Compute energies
    E_exact = duffing.energy(q_exact, v_exact)
    E_leapfrog = duffing.energy(q_leapfrog, v_leapfrog)
    E_imp_midpoint = duffing.energy(q_imp_midpoint, v_imp_midpoint)
    E_mid_dg = duffing.energy(q_mid_dg, v_mid_dg)
    E_mean_dg = duffing.energy(q_mean_dg, v_mean_dg)
   
    # Create plots
    plt.figure(figsize=(16, 8))
    # Position plot
    plt.subplot(2, 2, 1)
    plt.plot(t, q_exact, 'k--', label='Exact', linewidth=2)
    plt.plot(t_leapfrog, q_leapfrog, label='Leapfrog')
    plt.plot(t_imp_midpoint, q_imp_midpoint, label='Implicit Midpoint')
    plt.plot(t_mid_dg, q_mid_dg, label='Midpoint DG')
    plt.plot(t_mean_dg, q_mean_dg, label='Mean Value DG')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.title('Position vs Time')

    # Position error
    plt.subplot(2, 2, 2)
    plt.plot(t_leapfrog, np.abs(q_leapfrog - q_exact), label='Leapfrog')
    plt.plot(t_imp_midpoint, np.abs(q_imp_midpoint - q_exact), label='Implicit Midpoint')
    plt.plot(t_mid_dg, np.abs(q_mid_dg - q_exact), label='Midpoint DG')
    plt.plot(t_mean_dg, np.abs(q_mean_dg - q_exact), label='Mean Value DG')
    plt.xlabel('Time')
    plt.ylabel('Position Error')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title('Position Error vs Time')

    # Velocity plot
    plt.subplot(2, 2, 3)
    plt.plot(t, v_exact, 'k--', label='Exact', linewidth=2)
    plt.plot(t_leapfrog, v_leapfrog, label='Leapfrog')
    plt.plot(t_imp_midpoint, v_imp_midpoint, label='Implicit Midpoint')
    plt.plot(t_mid_dg, v_mid_dg, label='Midpoint DG')
    plt.plot(t_mean_dg, v_mean_dg, label='Mean Value DG')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    plt.title('Velocity vs Time')

    # Velocity error
    plt.subplot(2, 2, 4)
    plt.plot(t_leapfrog, np.abs(v_leapfrog - v_exact), label='Leapfrog')
    plt.plot(t_imp_midpoint, np.abs(v_imp_midpoint - v_exact), label='Implicit Midpoint')
    plt.plot(t_mid_dg, np.abs(v_mid_dg - v_exact), label='Midpoint DG')
    plt.plot(t_mean_dg, np.abs(v_mean_dg - v_exact), label='Mean Value DG')
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
    plt.plot(q_leapfrog, v_leapfrog, label='Leapfrog')
    plt.plot(q_imp_midpoint, v_imp_midpoint, label='Implicit Midpoint')
    plt.plot(q_mid_dg, v_mid_dg, label='Midpoint DG')
    plt.plot(q_mean_dg, v_mean_dg, label='Mean Value DG')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    plt.title('Phase Space')
    
    # Energy error
    plt.subplot(1, 2, 2)
    plt.plot(t_leapfrog, np.abs(E_leapfrog - E_exact[0])/E_exact[0], label='Leapfrog')
    plt.plot(t_imp_midpoint, np.abs(E_imp_midpoint - E_exact[0])/E_exact[0], label='Implicit Midpoint')
    plt.plot(t_mid_dg, np.abs(E_mid_dg - E_exact[0])/E_exact[0], label='Midpoint DG')
    plt.plot(t_mean_dg, np.abs(E_mean_dg - E_exact[0])/E_exact[0], label='Mean Value DG')
    plt.xlabel('Time')
    plt.ylabel('Relative Energy Error')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title('Energy Error vs Time')
    
    plt.show()

if __name__ == "__main__":
    compare_all_methods()