import firedrake as fdrk
from .problem import DynamicProblem
from src.meshing.cook_membrane import create_cook_membrane
import matplotlib.pyplot as plt
from src.tools.elasticity import compliance

class DynamicCookMembrane(DynamicProblem):

    def __init__(self, mesh_size):

        create_cook_membrane(mesh_size)
        self.domain = fdrk.Mesh('cook_membrane.msh')
        self.dim = self.domain.topological_dimension()

        # fig, axes = plt.subplots()
        # fdrk.triplot(self.domain, axes=axes)
        # axes.legend()
        # plt.show()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)

        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.parameters = {"rho": 1, 
                           "E": 250, 
                           "nu": 0.3}

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
        magnitude_traction = 6.5
        # t_coutoff_forcing = fdrk.Constant(5)
        # traction_y = time_nat/t_coutoff_forcing *  \
        # fdrk.conditional(fdrk.le(time_nat, t_coutoff_forcing), magnitude_traction, 0)

        traction = fdrk.as_vector([fdrk.Constant(0), magnitude_traction])

        return {3: traction, "follower": True}



    def get_forcing(self, time: fdrk.Constant):
        return None
    

    def __str__(self):
        return "DynamicCookMembrane"