import firedrake as fdrk
from .problem import StaticProblem
from src.tools.von_karman import bending_stiffness
from math import pi
class Wrinkling(StaticProblem):

    def __init__(self, n_elem_x, n_elem_y, thickness = 0.01):

        self.domain = fdrk.RectangleMesh(n_elem_x, n_elem_y, 1, 0.5, originX=0, originY=-0.5)

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)

        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)

        self.parameters = {"E": 1, "nu": 0, "h": thickness}
        
    

    def first_piola_definition(self, grad_disp):
        pass


    def second_piola_definition(self, cauchy_strain):
        pass


    def derivative_first_piola(self, tensor, grad_disp):
        pass


    def get_initial_conditions(self):
        u_0 = fdrk.as_vector([0, -self.y/10])
        w_0 = 0.5*self.x*(1 - self.x)**2*fdrk.sin(4*pi*self.y)
        kappa_0 = fdrk.grad(fdrk.grad(w_0))
        M_0 = bending_stiffness(kappa_0, self.parameters)

        return {"membrane displacement": u_0,
                "bending displacement": w_0, 
                "bending stress": M_0}


    def get_essential_bcs(self) -> dict:
        """
        Cantilever beam
        Zero velocity on left boundary 
        
        """

        essential_dict = {"displacement x": {1: fdrk.Constant(0)},
                          "displacement y": {1: -self.y/10},
                          "displacement z": {1: fdrk.Constant(0)}, 
                          "bending stress" : {2 : fdrk.Constant(((0.0, 0.0), (0.0, 0.0))), 
                                            3 : fdrk.Constant(((0.0, 0.0), (0.0, 0.0))),
                                            4 : fdrk.Constant(((0.0, 0.0), (0.0, 0.0)))}}
        
        return essential_dict
    
    def get_natural_bcs(self) -> dict:

        return {"traction x": {2: fdrk.Constant((0, 0)), 3: fdrk.Constant((0, 0)), 4: fdrk.Constant((0, 0))},
                "traction y": {2: fdrk.Constant((0, 0)), 3: fdrk.Constant((0, 0)), 4: fdrk.Constant((0, 0))}}


    def get_forcing(self):
        return None
    

    def __str__(self):
        return "Wrinkling"