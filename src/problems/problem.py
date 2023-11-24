import firedrake as fdrk
from abc import ABC, abstractmethod
from math import pi

class Problem(ABC):
    def __init__(self):
        self.domain = None
        self.coordinates_mesh = None
        self.bc_type = None
        self.forcing = None
        self.dim = None
        self.normal_versor = None
        self.quad = None
        self.parameters = None



    @abstractmethod
    def get_initial_conditions(self):
        pass


    @abstractmethod
    def get_forcing(self, time: fdrk.Constant):
        pass

    
    @abstractmethod
    def get_boundary_conditions(self, time_ess: fdrk.Constant, time_nat: fdrk.Constant):
        pass
