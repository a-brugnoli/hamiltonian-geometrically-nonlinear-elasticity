import firedrake as fdrk
from tqdm import tqdm
import numpy as np

class VonKarmanBeam:
    def __init__(self, physical_parameters = {"rho":1, "E": 1, "A": 1, "I":1, "L":1},
                simulation_parameters = {"time_step": 0.01, "t_span": np.array([0, 1]),\
                                          "n_elem": 30}, 
                ):
        
        rho = physical_parameters["rho"]
        E = physical_parameters["E"]
        A = physical_parameters["A"]
        I = physical_parameters["I"]
        self.density = rho*A
        self.bending_stiffness = E*I
        self.axial_stiffness = E*A
        self.bending_compliance = 1/self.bending_stiffness
        self.axial_compliance = 1/self.axial_stiffness
        self.length = physical_parameters["L"]

        self.n_elem = simulation_parameters["n_elem"]
        self.time_step = simulation_parameters["time_step"]

        t_span = simulation_parameters["t_span"]
        simulation_time = t_span[1] - t_span[0]
        self.n_steps =np.round(simulation_time/self.time_step).astype(int)
        T_init = t_span[0]
        T_end = self.n_steps*self.time_step + T_init
        self.t_span = np.array([T_init, T_end])
        self.t_vec = np.linspace(T_init, T_end, self.n_steps+1)

        self.domain = fdrk.IntervalMesh(self.n_elem, self.length)
        self.set_spaces()
        

    def set_spaces(self):
        
        self.space_q_x = fdrk.FunctionSpace(self.domain, "CG", 1)
        self.space_q_z = fdrk.FunctionSpace(self.domain, "Hermite", 3)

        self.space_v_x = self.space_q_x
        self.space_v_z = self.space_q_z

        self.space_axial_stress = fdrk.FunctionSpace(self.domain, "DG", 0)
        self.space_bending_stress = fdrk.FunctionSpace(self.domain, "DG", 1)

        self.mixed_space_dis_gradient = self.space_q_x * self.space_q_z * self.space_v_x * self.space_v_z 
        self.mixed_space_lin_implicit = self.space_v_x * self.space_v_z * self.space_axial_stress * self.space_bending_stress

        
    def set_initial_conditions(self, \
            initial_conditions = {"q_x": fdrk.Constant(0), \
                                    "q_z": fdrk.Constant(0), \
                                    "v_x": fdrk.Constant(0), \
                                    "v_z": fdrk.Constant(0)}):

        
        q_x_exp = initial_conditions["v_x"]
        q_z_exp = initial_conditions["v_z"]
 
        v_x_exp = initial_conditions["v_x"]
        v_z_exp = initial_conditions["v_z"]

        q_x_t0 = fdrk.interpolate(q_x_exp, self.space_q_x)
        q_z_t0 = fdrk.interpolate(q_z_exp, self.space_q_z)

        v_x_t0 = fdrk.interpolate(v_x_exp, self.space_v_x)
        v_z_t0 = fdrk.interpolate(v_z_exp, self.space_v_z)

        return q_x_t0, q_z_t0, v_x_t0, v_z_t0


    def axial_strain(self, q_x, q_z):
        return q_x.dx(0) + 1/2*(q_z.dx(0))**2

    def bending_strain(self, q_z):
        return q_z.dx(0).dx(0)
    
    def axial_stress(self, q_x, q_z):
        return self.axial_stiffness*self.axial_strain(q_x, q_z)
    
    def bending_stress(self, q_z):
        return self.bending_stiffness*self.bending_strain(q_z)
    

    def deformation_energy(self, q_x, q_z):
        potential_energy_density = 0.5*(self.axial_stiffness*self.axial_strain(q_x, q_z)**2 \
                                    + self.bending_stiffness*self.bending_strain(q_z)**2)
        potential_energy = fdrk.assemble(potential_energy_density*fdrk.dx)

        return potential_energy

    def kinetic_energy(self, v_x, v_z):
        kinetic_energy_density = 0.5*self.density*(v_x**2 + self.density*v_z**2)
        kinetic_energy = fdrk.assemble(kinetic_energy_density*fdrk.dx)

        return kinetic_energy

    def hamiltonian(self, q_x, q_z, v_x, v_z):
        energy_density = self.kinetic_energy(v_x, v_z) + self.deformation_energy(q_x, q_z)
        energy = fdrk.assemble(energy_density*fdrk.dx)
        return energy