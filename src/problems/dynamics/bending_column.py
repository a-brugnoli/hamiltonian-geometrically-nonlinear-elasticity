import firedrake as fdrk
from ..problem import DynamicProblem
from math import pi
import matplotlib.pyplot as plt
import numpy as np


class BendingColumn(DynamicProblem):

    def __init__(self, n_elem_x, n_elem_y, n_elem_z):

        self.domain = fdrk.BoxMesh(n_elem_x, n_elem_y, n_elem_z, Lx=1, Ly=1, Lz=6)
        self.dim = self.domain.topological_dimension()

        self.domain.coordinates.dat.data[:, 0] -= 0.5
        self.domain.coordinates.dat.data[:, 1] -= 0.5

        offset = 5.2 * pi/180
        angle = pi/4 - offset
        rotated_x_coordinate = np.cos(angle) * self.domain.coordinates.dat.data[:, 0] \
                            - np.sin(angle) * self.domain.coordinates.dat.data[:, 1]
        
        rotated_y_coordinate = np.sin(angle) * self.domain.coordinates.dat.data[:, 0] \
                             + np.cos(angle) * self.domain.coordinates.dat.data[:, 1]

        self.domain.coordinates.dat.data[:, 0] = rotated_x_coordinate
        self.domain.coordinates.dat.data[:, 1] = rotated_y_coordinate
        
        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # fdrk.triplot(self.domain, axes=axes)
        # axes.legend()
        # plt.show()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)
        self.x, self.y, self.z = self.coordinates_mesh

        
        self.normal_versor = fdrk.FacetNormal(self.domain)

        young_modulus = 17*10**6
        poisson_ratio = 0.3

        mu = young_modulus / (2*(1 + poisson_ratio))
        lamda = young_modulus*poisson_ratio/((1 - 2*poisson_ratio)*(1 + poisson_ratio))
        kappa = lamda + 2/3*mu

        self.parameters = {"rho": 1.1*10**3, # kg/m^3 
                        "E": young_modulus, 
                        "nu":poisson_ratio,
                        "mu": mu, 
                        "kappa": kappa, \
                        "lamda": lamda}

    def get_initial_conditions(self):

        velocity_0 =fdrk.as_vector([5/3*self.z, 0, 0])

        displacement_0 = fdrk.as_vector([0, 0, 0])
        strain_0 = fdrk.as_tensor([[0, 0, 0], 
                                   [0, 0, 0],
                                   [0, 0, 0]])

        return {"displacement": displacement_0,
                "velocity": velocity_0, 
                "strain": strain_0
                }


    def get_essential_bcs(self, time_ess):
        """
        Cantilever beam
        Zero velocity on left boundary 
        Traction along the y axis on the right boundary =
        
        """
        essential_dict = {"displacement": {5: fdrk.as_vector([0, 0, 0])}, \
                        "velocity": {5: fdrk.as_vector([0, 0, 0])}}
        
        return essential_dict


    def get_natural_bcs(self, time_nat):
        return None



    def get_forcing(self, time: fdrk.Constant):
        return None
    

    def __str__(self):
        return "BendingColumn"