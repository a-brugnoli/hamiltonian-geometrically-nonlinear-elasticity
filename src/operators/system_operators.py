import firedrake as fdrk
from src.problems.problem import Problem
from abc import ABC, abstractmethod


class DynamicalOperators(ABC):
    def __init__(self, problem: Problem, pol_degree):
        """
        Constructor for the MaxwellOperators class
        Parameters
            type (string) : "primal" or "dual", the kind of discretization (primal is u1 or B2)
            reynold (float) : the reciprocal of the magnetic Reynolds number
        """
        self.problem = problem
        self.domain = problem.domain
        self.pol_degree = pol_degree
        self.normal_versor = fdrk.FacetNormal(self.domain)
        self.cell_name = str(self.domain.ufl_cell())

        self._set_space()


    @abstractmethod
    def _set_space():
        pass


    @abstractmethod
    def mass_def_gradient():
        pass


    @abstractmethod
    def dynamics_def_gradient():
        pass


    def operator_def_gradient(self, testfunction, trialfunction):
        """
        Construct the mass matrix for the displacement part (for explicit time integration)
        """
        mass_operator = self.mass_def_gradient(testfunction, trialfunction)

        return mass_operator
    

    def functional_def_gradient(self, time_step, testfunction, def_gradient_old, velocity):
        """
        Construct the functional associated with the explicit time integration of the displacement part
        b = ()
        """
        rhs_functional = self.mass_def_gradient(testfunction, def_gradient_old)\
                     + time_step * self.dynamics_def_gradient(testfunction, velocity)
        
        return rhs_functional


    @abstractmethod
    def mass_energy():
        pass


    @abstractmethod
    def dynamics_energy():
        pass


    @abstractmethod
    def control():
        pass


    def operator_energy(self, time_step, testfunctions, trialfunctions, displacement):
        """
        Construct operators arising from the implicit midpoint discretization of
        the energy part of the system

        A = M - 0.5 * dt *  J(d)
        """
        mass_operator = self.mass_energy(testfunctions, trialfunctions)
        dynamics_operator = self.dynamics_energy(testfunctions, trialfunctions, displacement)

        lhs_operator = mass_operator - 0.5 * time_step * dynamics_operator
        
        return lhs_operator
    

    def functional_energy(self, time_step, testfunctions, old_states, def_gradient, control_midpoint_dict):
        """
        Construct functional arising from the implicit midpoint discretization of
        the energy part of the system
        b = ( M + 0.5 * dt * J(d_midpoint) ) x_old + B u_midpoint
        """

        mass_functional = self.mass_energy(testfunctions, old_states)
        dynamics_functional = self.dynamics_energy(testfunctions, old_states, def_gradient)

        natural_control = self.control(testfunctions, def_gradient, control_midpoint_dict)

        rhs_functional = mass_functional + 0.5 * time_step * dynamics_functional \
                                    + time_step * natural_control
        
        return rhs_functional
    

    def __str__(self) -> str:
        return f"SystemOperatorAbstractClass"