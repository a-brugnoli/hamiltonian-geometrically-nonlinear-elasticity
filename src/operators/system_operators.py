import firedrake as fdrk
from src.problems.problem import Problem
from abc import ABC, abstractmethod

class SystemOperators(ABC):
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
        self.cell_diameter = fdrk.CellDiameter(self.domain)
        self.cell_name = str(self.domain.ufl_cell())

        self._set_space()


    @abstractmethod
    def _set_space():
        pass

    @abstractmethod
    def get_initial_conditions():
        pass


    @abstractmethod
    def essential_boundary_conditions():
        pass

    @abstractmethod
    def natural_boundary_conditions():
        pass


    @abstractmethod
    def mass_displacement():
        pass


    @abstractmethod
    def dynamics_displacement():
        pass



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
        Construct operators arising from the implicit midpoint discretization
        A x = b
        """
        mass_operator = self.mass_energy(testfunctions, trialfunctions)
        dynamics_operator = self.dynamics_energy(testfunctions, trialfunctions)

        lhs_operator = mass_operator - 0.5 * time_step * dynamics_operator
        
        return lhs_operator
    

    def functional_energy(self, time_step, testfunctions, functions, control):

        mass_functional = self.mass_energy(testfunctions, functions)
        dynamics_functional = self.dynamics_energy(testfunctions, functions)

        natural_control = self.control(testfunctions, control)

        rhs_functional = mass_functional + 0.5 * time_step * dynamics_functional \
                                    + time_step * natural_control
        
        return rhs_functional
    
    def __str__(self) -> str:
        return f"SystemOperatorAbstractClass"