import firedrake as fdrk
from abc import ABC, abstractmethod
from math import pi

class Problem(ABC):
    def __init__(self):
        self.domain = None
        self.dim = None
        self.coordinates_mesh = None
        self.x, self.y, self.z = None, None, None
        self.bc_type = None
        self.forcing = None
        self.dim = None
        self.normal_versor = None
        self.quad = None
        self.parameters = None


class StaticProblem(Problem):
    def __init__(self):
        super().__init__()


    @abstractmethod
    def second_piola_definition(self):
        pass


    @abstractmethod
    def first_piola_definition(self):
        pass


    @abstractmethod
    def derivative_first_piola(self):
        pass

    @abstractmethod
    def get_forcing(self):
        pass

    
    @abstractmethod
    def get_essential_bcs(self) -> dict:
        pass


    @abstractmethod
    def get_natural_bcs(self) -> dict:
        pass


class DynamicProblem(Problem):
    def __init__(self):
        super().__init__()


    @abstractmethod
    def get_forcing(self, time: fdrk.Constant):
        pass

    
    @abstractmethod
    def get_essential_bcs(self, time_ess: fdrk.Constant) -> dict:
        pass


    @abstractmethod
    def get_natural_bcs(self, time_nat: fdrk.Constant) -> dict:
        pass