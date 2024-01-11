import firedrake as fdrk
from .problem import StaticProblem

# Neo Hookean Potentials
# I_1, I_2, I_3 are the principal invariants of the Cauchy Green deformation tensor C = F^T F
# W_1 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/2 * (I_3^(1/2) - 1)^2
# W_2 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/8 *ln(I_3)^2

# First Piola stress tensor
# P_1 = mu (F - F^{-T}) + kappa (J^2 - J) F^{-T}
# P_2 = mu (F - F^{-T}) + kappa ln(J) F^{-T}

class ConvergenceStatic(StaticProblem):

    def __init__(self, n_elem_x, n_elem_y):

        self.domain = fdrk.UnitSquareMesh(n_elem_x, n_elem_y)
        self.dim = self.domain.geometric_dimension()

        # fig, axes = plt.subplots()
        # fdrk.triplot(self.domain, axes=axes)
        # axes.legend()
        # plt.show()

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)

        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.mu = 1  #N/mm^2
        self.kappa = 1 #N/mm^2


    def first_piola_definition(self, grad_disp):
        def_grad = fdrk.Identity(self.dim) + grad_disp
        inv_F_transpose = fdrk.inv(def_grad).T
        return self.mu*(def_grad - inv_F_transpose) + self.kappa * fdrk.ln(fdrk.det(def_grad)) * inv_F_transpose


    def derivative_first_piola(self, tensor, grad_disp):
        def_grad = fdrk.Identity(self.dim) + grad_disp
        invF = fdrk.inv(def_grad)
        inv_Ftr = fdrk.inv(def_grad).T

        return self.mu * tensor + (self.mu - self.kappa * fdrk.ln(fdrk.det(def_grad))) \
                * fdrk.dot(inv_Ftr, fdrk.dot(tensor.T, inv_Ftr)) \
                + self.kappa * fdrk.tr(fdrk.dot(invF, tensor)) * inv_Ftr


    def get_exact_solution(self):

        exact_displacement = fdrk.as_vector([0.5*self.y**3 + 0.5*fdrk.sin(0.5 * fdrk.pi * self.y), 
                                             fdrk.Constant(0)])
        
        exact_disp_grad = fdrk.grad(exact_displacement)
        exact_first_piola = self.first_piola_definition(exact_disp_grad)

        return {"displacement" : exact_displacement,
                "disp_grad": exact_disp_grad,
                "first_piola": exact_first_piola}


    def get_essential_bcs(self) -> dict:
        """
        Cantilever beam
        Zero velocity on left boundary 
        Traction along the y axis on the right boundary =
        
        """

        essential_dict = {"displacement x": {3: fdrk.Constant(0)},
                          "displacement y": {3: fdrk.Constant(0)}}
        
        return essential_dict
    

    def get_natural_bcs(self) -> dict:

        first_piola_exact = self.get_exact_solution()["first_piola"]

        traction = fdrk.dot(first_piola_exact, self.normal_versor)

        # traction_1 = fdrk.as_vector(-first_piola_exact[:, 0])
        # traction_2 = fdrk.as_vector(+first_piola_exact[:, 0])
        # traction_4 = fdrk.as_vector(+first_piola_exact[:, 1])

        return {"traction x": {1: traction[0], 2: traction[0], 4: traction[0]},
                "traction y": {1: traction[1], 2: traction[1], 4: traction[1]}}


    def get_forcing(self):

        first_piola_exact = self.get_exact_solution()["first_piola"]
        exact_forcing = - fdrk.div(first_piola_exact)
        return exact_forcing
    
    

    def __str__(self):
        return "ConvergenceStatic2D"