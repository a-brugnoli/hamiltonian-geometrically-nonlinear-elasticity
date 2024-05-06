import firedrake as fdrk
from .problem import Problem
import numpy as np

class CantileverBeam(Problem):

    def __init__(self, n_elem_x, n_elem_y, quad=False):
        L_x = 100
        L_y = 10

        self.domain = fdrk.RectangleMesh(n_elem_x, n_elem_y, L_x, L_y, originX=0, originY=0, quadrilateral=quad)

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)
        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        density = fdrk.Constant(1)
        young_modulus = fdrk.Constant(1000)
        poisson_ratio = fdrk.Constant(0.3)

        mu = young_modulus / (2*(1 + poisson_ratio))
        lamda = young_modulus*poisson_ratio/((1 - 2*poisson_ratio)*(1 + poisson_ratio))
        kappa = lamda + 2/3*mu

        self.parameters = {"E": young_modulus, 
                            "nu": poisson_ratio,
                            "rho": density, # kg/m^3 
                            "lamda":lamda,
                            "mu": mu, 
                            "kappa": kappa}

    def get_initial_conditions(self):
        velocity_0 = fdrk.as_vector([0.0, 0.0])
        stress_0 = fdrk.as_vector([[0, 0], [0, 0]])

        displacement_0 = fdrk.as_vector([0.0, 0.0])

        return {"displacement": displacement_0,
                "velocity": velocity_0, 
                "stress": stress_0, 
                }
    

    def get_essential_bcs(self, time_ess):
        """
        Cantilever beam
        Zero velocity on left boundary 
        Traction along the y axis on the right boundary =
        
        """
        essential_dict = {"displacement": {1: fdrk.as_vector([0, 0])}, \
                        "velocity": {1: fdrk.as_vector([0, 0])}}
        
        return essential_dict
    
    def get_natural_bcs(self, time_nat):
        t_coutoff_forcing = fdrk.Constant(5)
        magnitude_traction = 50

        traction_y = time_nat/t_coutoff_forcing *  \
        fdrk.conditional(fdrk.le(time_nat, t_coutoff_forcing), magnitude_traction, 0)
        traction = fdrk.as_vector([fdrk.Constant(0), traction_y])

        return {2: traction, "follower": True}




    def get_forcing(self, time: fdrk.Constant):
        return None
    

    def __str__(self):
        return "CantileverBeam"