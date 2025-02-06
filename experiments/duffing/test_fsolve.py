# main_solver.py
import numpy as np
from scipy.optimize import fsolve
from external_module import calculate_parameter

def equation_to_solve(t, pressure, temperature):
    """
    Base equation that depends on the external function
    We want to find t where this equation equals zero
    """
    # Get parameter from external function
    param = calculate_parameter(t, pressure, temperature)
    
    # Our equation: param * t^2 - 100 = 0
    return param * t**2 - 100

def create_solver(pressure, temperature):
    """
    Creates a wrapper function for fsolve that includes our parameters
    """
    def wrapper(t):
        return equation_to_solve(t, pressure, temperature)
    return wrapper

def find_solution(initial_guess, pressure, temperature):
    """
    Main solving function that uses fsolve
    """
    # Create the wrapper function with our parameters
    solver_function = create_solver(pressure, temperature)
    
    # Use fsolve to find the root
    solution = fsolve(solver_function, initial_guess)[0]
    
    # Verify the solution
    verification = equation_to_solve(solution, pressure, temperature)
    
    return solution, verification

def main():
    # Example parameters
    pressure = 1.0      # Some pressure value
    temperature = 300   # Some temperature value
    initial_guess = 1.0 # Initial guess for solver
    
    # Find the solution
    solution, verification = find_solution(initial_guess, pressure, temperature)
    
    print(f"Solution found: t = {solution:.6f}")
    print(f"Verification (should be close to 0): {verification:.10f}")
    
    # Get the final parameter value at the solution
    final_param = calculate_parameter(solution, pressure, temperature)
    print(f"Final parameter value: {final_param:.6f}")
    
    # Try with different parameters
    pressure = 2.0
    temperature = 400
    solution, verification = find_solution(initial_guess, pressure, temperature)
    
    print(f"\nWith different parameters (P={pressure}, T={temperature}):")
    print(f"Solution found: t = {solution:.6f}")
    print(f"Verification (should be close to 0): {verification:.10f}")

if __name__ == "__main__":
    main()