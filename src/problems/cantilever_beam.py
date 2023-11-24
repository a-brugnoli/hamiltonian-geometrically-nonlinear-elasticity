import firedrake as fdrk
from .problem import Problem
import numpy as np

class CantileverBeam(Problem):

    def __init__(self, n_elem_x, n_elem_y, quad=False):
        L_x = 100
        L_y = 10

        self.domain = fdrk.RectangleMesh(n_elem_x, n_elem_y, L_x, L_y, quadrilateral=quad)

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)
        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        h = 1
        density = fdrk.Constant(1)
        young_modulus = fdrk.Constant(1000)
        poisson_modulus = fdrk.Constant(0.3)

        self.parameters = {"Density": density, 
                           "Young modulus": young_modulus, 
                           "Poisson modulus": poisson_modulus, 
                           "Thickness": h}


    def get_initial_conditions(self):
        def_gradient_0 = fdrk.Constant(np.array([[1, 0], [0, 1]]))
        velocity_0 = fdrk.Constant(np.array([0, 0]))
        stress_0 = fdrk.Constant(np.array([[0, 0], [0, 0]]))

        displacement_0 = fdrk.Constant(np.array([0, 0]))

        return {"def_gradient": def_gradient_0, 
                "velocity": velocity_0, 
                "stress": stress_0, 
                "displacement": displacement_0}
    

    def get_boundary_conditions(self, time_ess, time_nat):
        """
        Cantilever beam
        Zero velocity on left boundary 
        Traction along the y axis on the right boundary =
        
        """
        
        t_coutoff_forcing = fdrk.Constant(5)
        magnitude_traction = 50

        traction_y = time_nat/t_coutoff_forcing *  \
        fdrk.conditional(fdrk.le(time_nat, t_coutoff_forcing), magnitude_traction, 0)
        traction = fdrk.as_vector([fdrk.Constant(0), traction_y])

        bd_cond_dict = {"velocity": {1: fdrk.Constant(np.array([0, 0]))}, \
                        "traction" : {2: traction}}
        
        return bd_cond_dict


    def get_forcing(self, time: fdrk.Constant):
        return None