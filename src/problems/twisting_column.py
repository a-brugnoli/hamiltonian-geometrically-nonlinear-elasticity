import firedrake as fdrk
from .problem import DynamicProblem
from math import pi
import matplotlib.pyplot as plt

class TwistingColumn(DynamicProblem):

    def __init__(self, n_elem_x, n_elem_y, n_elem_z):

        self.domain = fdrk.BoxMesh(n_elem_x, n_elem_y, n_elem_z, Lx=1, Ly=1, Lz=6)
        self.dim = self.domain.geometric_dimension()

        # fig = plt.figure()
        # axes = fig.add_subplot(111, projection='3d')
        # fdrk.triplot(self.domain, axes=axes)
        # axes.legend()
        # plt.show()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)

        self.x, self.y, self.z = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.parameters = {"rho": 1.1*10**3, # kg/m^3 
                           "E": 17*10**6, 
                           "nu": 0.3}

    def get_initial_conditions(self):

        omega = 100 # rag/s
        angular_velocity_0 = fdrk.as_vector([0, 0, omega*fdrk.sin(pi *self.z/12)])
        velocity_0 = fdrk.cross(angular_velocity_0, fdrk.as_vector([self.x, self.y, self.z]))

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
        return "TwisitingColumn"