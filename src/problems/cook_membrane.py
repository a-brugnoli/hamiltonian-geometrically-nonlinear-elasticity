import firedrake as fdrk
from .problem import StaticProblem
import numpy as np
from src.meshing.cook_membrane import create_cook_membrane
import matplotlib.pyplot as plt

# Neo Hookean Potentials
# I_1, I_2, I_3 are the principal invariants of the Cauchy Green deformation tensor C = F^T F
# W_1 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/2 * (I_3^(1/2) - 1)^2
# W_2 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/8 *ln(I_3)^2

# First Piola stress tensor
# P_1 = mu (F - F^{-T}) + kappa (J^2 - J) F^{-T}
# P_2 = mu (F - F^{-T}) + kappa ln(J) F^{-T}

class CookMembrane(StaticProblem):

    def __init__(self, mesh_size):

        create_cook_membrane(mesh_size)
        self.domain = fdrk.Mesh('cook_membrane.msh')
        self.dim = self.domain.geometric_dimension()

        # fig, axes = plt.subplots()
        # fdrk.triplot(self.domain, axes=axes)
        # axes.legend()
        # plt.show()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)

        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.mu = 80.194  #N/mm^2
        self.kappa = 400889.8 #N/mm^2


    def energy_definition(self, grad_disp):
        def_grad = fdrk.Identity(self.dim) + grad_disp
        Cauchy_strain = fdrk.dot(def_grad.T, def_grad) 
        J = fdrk.det(def_grad)
        W = self.mu/2 (fdrk.tr(Cauchy_strain) - 3) - self.mu * fdrk.ln(J) + self.kappa/2 * (fdrk.ln(J))**2
        return W



    def first_piola_definition(self, grad_disp):
        def_grad = fdrk.Identity(self.dim) + grad_disp
        inv_F_transpose = fdrk.inv(def_grad.T)
        J = fdrk.det(def_grad)
        return self.mu*(def_grad - inv_F_transpose) + self.kappa * fdrk.ln(J) * inv_F_transpose


    def derivative_first_piola(self, tensor, grad_disp):
        def_grad = fdrk.Identity(self.dim) + grad_disp
        invF = fdrk.inv(def_grad)
        inv_Ftr = fdrk.inv(def_grad.T)

        return self.mu * tensor + (self.mu - self.kappa * fdrk.ln(fdrk.det(def_grad))) \
                * fdrk.dot(inv_Ftr, fdrk.dot(tensor.T, inv_Ftr)) \
                + self.kappa * fdrk.tr(fdrk.dot(invF, tensor)) * inv_Ftr


    def get_essential_bcs(self) -> dict:
        """
        Cantilever beam
        Zero velocity on left boundary 
        Traction along the y axis on the right boundary =
        
        """

        essential_dict = {"displacement x": {1: fdrk.Constant(0)},
                          "displacement y": {1: fdrk.Constant(0)}}
        
        return essential_dict
    
    def get_natural_bcs(self) -> dict:


        force_y = 8
        # traction = fdrk.as_vector([fdrk.Constant(0), force_y])

        return {"traction x": {2: fdrk.Constant(0), 3: fdrk.Constant(0), 4: fdrk.Constant(0)},
                "traction y": {2: fdrk.Constant(0), 3: force_y, 4: fdrk.Constant(0)}}


    def get_forcing(self):
        return None
    

    def __str__(self):
        return "CookMembrane"