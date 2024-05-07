import firedrake as fdrk
from ..problem import StaticProblem
import numpy as np

# Neo Hookean Potentials
# I_1, I_2, I_3 are the principal invariants of the Cauchy Green deformation tensor C = F^T F
# W_1 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/2 * (I_3^(1/2) - 1)^2
# W_2 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/8 *ln(I_3)^2

# First Piola stress tensor
# P_1 = mu (F - F^{-T}) + kappa (J^2 - J) F^{-T}
# P_2 = mu (F - F^{-T}) + kappa ln(J) F^{-T}

class InhomogeneousCompression(StaticProblem):

    def __init__(self, n_elem_x, n_elem_y, quad=False):
        self.length_side = 10 #mm

        self.domain = fdrk.RectangleMesh(n_elem_x, n_elem_y, 
                                         Lx = self.length_side, 
                                         Ly = self.length_side, 
                                         quadrilateral=quad)
        
        self.dim = self.domain.topological_dimension()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)
        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        mu = 80.194  #N/mm^2
        lamda = 400889.8 #N/mm^2

        self.parameters = {"mu": mu, "lamda": lamda}

    

    def get_essential_bcs(self) -> dict:
        """
        Cantilever beam
        Zero velocity on left boundary 
        Traction along the y axis on the right boundary =
        
        """

        essential_dict = {"displacement x": {1: fdrk.Constant(0), 4: fdrk.Constant(0)},
                          "displacement y": {3: fdrk.Constant(0)}}
        
        return essential_dict
    
    def get_natural_bcs(self) -> dict:

        factor = 1
        f = 600
        force_y = -factor*f*fdrk.conditional(fdrk.le(self.x, self.length_side/2), 1, 0) 
        traction = fdrk.as_vector([fdrk.Constant(0), force_y])

        return {"traction x": {2: fdrk.Constant((0, 0)), 3: fdrk.Constant((0, 0))},
                "traction y": {1: fdrk.Constant((0, 0)), 2: fdrk.Constant((0, 0)), 4: traction}}
    

    def get_forcing(self):
        return None
    

    def __str__(self):
        return "InhomogeneousCompression"