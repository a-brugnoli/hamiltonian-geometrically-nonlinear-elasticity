import firedrake as fdrk
from .problem import StaticProblem
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
                                         Ly = self.length_side)
        
        self.dim = self.domain.geometric_dimension()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)
        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.mu = 80.194  #N/mm^2
        self.lamda = 400889.8 #N/mm^2

    def first_piola_definition(self, grad_disp):
        def_grad = fdrk.Identity(self.dim) + grad_disp
        inv_F_transpose = fdrk.inv(def_grad).T
        return self.mu*(def_grad - inv_F_transpose) + self.lamda * fdrk.ln(fdrk.det(def_grad)) * inv_F_transpose


    def second_piola_definition(self, cauchy_strain):
        inv_cauchy_strain = fdrk.inv(cauchy_strain)
        return self.mu * (fdrk.Identity(self.dim) - inv_cauchy_strain) \
             + self.lamda/2 * fdrk.ln(fdrk.det(cauchy_strain))*inv_cauchy_strain
    

    def derivative_first_piola(self, tensor, grad_disp):
        def_grad = fdrk.Identity(self.dim) + grad_disp
        invF = fdrk.inv(def_grad)
        inv_Ftr = fdrk.inv(def_grad).T

        return self.mu * tensor + (self.mu - self.lamda * fdrk.ln(fdrk.det(def_grad))) \
                * fdrk.dot(inv_Ftr, fdrk.dot(tensor.T, inv_Ftr)) \
                + self.lamda * fdrk.tr(fdrk.dot(invF, tensor)) * inv_Ftr


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