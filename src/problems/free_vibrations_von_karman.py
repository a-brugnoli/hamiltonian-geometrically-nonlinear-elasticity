import firedrake as fdrk
from .problem import Problem
import numpy as np
from math import pi
from src.tools.von_karman import bending_stiffness

class FirstModeVonKarman(Problem):

    def __init__(self, n_elem_x, n_elem_y, amplitude = 0.01):
        L_x = 0.5
        L_y = 0.5

        self.domain = fdrk.RectangleMesh(n_elem_x, n_elem_y, L_x, L_y)

        self.coordinates_mesh = fdrk.SpatialCoordinate(self.domain)
        self.x, self.y = self.coordinates_mesh
        self.normal_versor = fdrk.FacetNormal(self.domain)


        self.amplitude = amplitude
        density = fdrk.Constant(7850) 
        thickness = 0.002
        young_modulus = fdrk.Constant(2 * 10**11)
        poisson_modulus = fdrk.Constant(0.3)

        self.parameters = {"rho": density, 
                           "h": thickness, 
                           "E": young_modulus, 
                           "nu": poisson_modulus, 
                           "L_x": L_x, 
                           "L_y": L_y}


    def get_initial_conditions(self):
        h = self.parameters["h"]

        L_x = self.parameters["L_x"]
        L_y = self.parameters["L_y"]

        bend_disp_0 = self.amplitude * h * fdrk.sin(pi * self.x/ L_x) * fdrk.sin(pi * self.y/ L_y)

        mem_velocity_0 = fdrk.Constant((0, 0))
        mem_stress_0 = fdrk.Constant(((0.0, 0.0), (0.0, 0.0)))

        bend_velocity_0 = fdrk.Constant(0)

        bending_strain_0 = fdrk.sym(fdrk.grad(fdrk.grad(bend_disp_0)))
        bend_stress_0 = bending_stiffness(bending_strain_0, self.parameters)

        return {"membrane velocity": mem_velocity_0,
                "membrane stress": mem_stress_0, 
                "bending displacement": bend_disp_0,
                "bending velocity": bend_velocity_0, 
                "bending stress": bend_stress_0
                }
    

    def get_essential_bcs(self, time_ess):
        """
        Simply supported boundary conditions
        
        """
        essential_dict = {"bending displacement": {"on_boundary": fdrk.Constant(0)}, \
                        "membrane velocity": {"on_boundary": fdrk.Constant((0, 0))}, \
                        "bending velocity": {"on_boundary": fdrk.Constant(0)}, \
                        "bending stress" : {"on_boundary": fdrk.Constant(((0.0, 0.0), (0.0, 0.0)))}}
        
        return essential_dict

    def get_natural_bcs(self, time_nat):

        return None


    def get_forcing(self, time: fdrk.Constant):
        return None
    

    def __str__(self):
        return "FirstModeVonKarman"