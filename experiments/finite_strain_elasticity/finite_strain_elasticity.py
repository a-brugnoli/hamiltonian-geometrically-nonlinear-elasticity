import firedrake as fdrk
import numpy as np
from tqdm import tqdm
from math import pi

class FiniteStrainElasticity:
    def __init__(self, **kwargs): 
        parameters = {
        "rho":1.1*10**3, "E": 17*10**6, "nu": 0.3, "Lx":1, "Ly":1, "Lz":6,
        "v0_z":5/3,  "n_elem": (3, 3, 18),  
        "time_step": 0.01, "t_span": np.array([0, 1]), "n_output": 100
        }

        for key, value in kwargs.items():
            if key in parameters:
                # print(f"Paramter {key} modified from default value {parameters[key]}")
                parameters[key] = value

        # Physical parameters
        self.density = parameters["rho"]
        self.E = parameters["E"]
        self.nu = parameters["nu"]
                
        # Initial conditions
        self.v0_z = parameters["v0_z"]

        # Mesh and space parameters
        nel_x, nel_y, nel_z = parameters["n_elem"]
        Lx = parameters["Lx"]
        Ly = parameters["Ly"]
        Lz = parameters["Lz"]
        self.domain = fdrk.BoxMesh(nel_x, nel_y, nel_z, Lx=Lx, Ly=Ly, Lz=Lz)
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
        
        self.coordinates = fdrk.SpatialCoordinate(self.domain)
        self.x_coord, self.y_coord, self.z_coord = fdrk.SpatialCoordinate(self.domain)

        DG0_space = fdrk.FunctionSpace(self.domain, 'DG', 0)
        diameters = fdrk.CellSize(self.domain)
        self.min_mesh_size = min(fdrk.assemble(fdrk.interpolate(diameters, DG0_space)).vector().get_local())

        # Time parameters
        time_step = parameters["time_step"]
        t_span = parameters["t_span"]
        self.n_output = parameters["n_output"]
        self.dt = fdrk.Constant(time_step)

        # Adjust final time to have exactly the right time step
        simulation_time = t_span[1] - t_span[0]
        self.n_steps =np.round(simulation_time/time_step).astype(int)
        T_init = t_span[0]
        T_end = self.n_steps*time_step + T_init
        self.t_span = np.array([T_init, T_end])
        self.t_vec = np.linspace(T_init, T_end, self.n_steps+1)

        # Create a vector of time for the output of size n_output
        n_sim_times = len(self.t_vec)
        if n_sim_times>self.n_output:
            n_steps_output = self.n_output-1
            self.output_frequency = int(np.floor(self.n_steps/n_steps_output))
        else:
            self.output_frequency = 1
        self.t_vec_output = self.t_vec[::self.output_frequency]

        # Finite element spaces
        self.space_displacement = fdrk.VectorFunctionSpace(self.domain, "CG", 1)
        self.space_velocity = self.space_displacement

        self.n_dofs_disp = int(self.space_displacement.dim()/3)


        # When non linear effects are strong
        # the degree for the axial variable needs to be high
        self.space_stress = fdrk.TensorFunctionSpace(self.domain, "DG", 0, symmetry=True)

        self.mixed_space_implicit = self.space_displacement * self.space_velocity
        self.mixed_space_linear_implicit = self.space_velocity * self.space_stress


    def set_time_step(self, time_step):
        self.dt.assign(time_step)
        # Adjust final time to have exactly the right time step
        simulation_time = self.t_span[1] - self.t_span[0]
        self.n_steps =np.round(simulation_time/time_step).astype(int)
        T_init = self.t_span[0]
        T_end = self.n_steps*time_step + T_init
        self.t_span = np.array([T_init, T_end])
        self.t_vec = np.linspace(T_init, T_end, self.n_steps+1)

        # Create a vector of time for the output of size n_output
        n_sim_times = len(self.t_vec)
        if n_sim_times>self.n_output:
            n_steps_output = self.n_output-1
            self.output_frequency = int(np.floor(self.n_steps/n_steps_output))
        else:
            self.output_frequency = 1
        self.t_vec_output = self.t_vec[::self.output_frequency]


    def get_initial_conditions(self):
        q0 = fdrk.as_vector([0, 0, 0])
        velocity_0 =fdrk.as_vector([5/3*self.z_coord, 0, 0])
        return {"displacement": q0, "velocity": velocity_0}


    def get_essential_bcs(self):
        """
        Cantilever beam
        Zero velocity on left boundary         
        """
        essential_dict = {"displacement": {5: fdrk.as_vector([0, 0, 0])}, \
                        "velocity": {5: fdrk.as_vector([0, 0, 0])}}
        return essential_dict


    def stiffness_tensor(self, strain_tensor):
        dim = strain_tensor.ufl_shape[0]

        stress_tensor = self.E/(1+self.nu)*\
                (strain_tensor + self.nu/(1-2*self.nu)*fdrk.Identity(dim)*fdrk.tr(strain_tensor))

        return stress_tensor 


    def compliance_tensor(self, stress_tensor):
        dim = stress_tensor.ufl_shape[0]
        # Compliance tensor for generic dimensions
        strain_tensor = 1 /self.E * ((1+self.nu)*stress_tensor \
                - self.nu * fdrk.Identity(dim) * fdrk.tr(stress_tensor))
        return strain_tensor
    

    def deformation_gradient(self, displacement):
        dim = displacement.ufl_shape[0]
        return fdrk.Identity(dim) + fdrk.grad(displacement)


    def green_lagrange_strain(self, displacement):
        return 1/2*(fdrk.grad(displacement).T + fdrk.grad(displacement) \
                + fdrk.dot(fdrk.grad(displacement).T, fdrk.grad(displacement)))


    def second_piola_stress(self, displacement):
        strain_tensor = self.green_lagrange_strain(displacement)
        return self.stiffness_tensor(strain_tensor)
        

    def infinitesimal_strain(self, displacement):
        return 1/2*(fdrk.grad(displacement).T + fdrk.grad(displacement))


    def deformation_energy(self, displacement):
        strain_tensor = self.green_lagrange_strain(displacement)
        stress_tensor = self.stiffness_tensor(strain_tensor)
        potential_energy_density = 0.5*fdrk.inner(stress_tensor, strain_tensor)
        return potential_energy_density*fdrk.dx


    def kinetic_energy(self, velocity):
        kinetic_energy_density = 0.5*self.density*fdrk.inner(velocity, velocity)
        return kinetic_energy_density*fdrk.dx


    def hamiltonian(self, displacement, velocity):
        return self.kinetic_energy(velocity) + self.deformation_energy(displacement)
    

    def deformation_energy_leapfrog(self, displacement_min, displacement_plus):
        strain_min = self.infinitesimal_strain(displacement_min)
        strain_plus = self.infinitesimal_strain(displacement_plus)
        deformation_density = 0.5*fdrk.inner(strain_min, self.stiffness_tensor(strain_plus))

        return deformation_density*fdrk.dx 
        

    def weak_grad_potential(self, test_velocity, displacement):
        def_gradient = self.deformation_gradient(displacement)
        second_piola_stress_tensor = self.second_piola_stress(displacement)

        form_dV = + fdrk.inner(fdrk.grad(test_velocity), fdrk.dot(def_gradient, second_piola_stress_tensor))*fdrk.dx 
        
        return form_dV
    

    def weak_grad_potential_stress(self, test_velocity, displacement, second_piola_stress_tensor):
        def_gradient = self.deformation_gradient(displacement)
        form_dV = + fdrk.inner(fdrk.grad(test_velocity), fdrk.dot(def_gradient, second_piola_stress_tensor))*fdrk.dx 

        return form_dV


    def weak_grad_potential_linear(self, test_velocity, displacement):
        infinitesimal_strain = self.infinitesimal_strain(displacement)
        cauchy_stress = self.stiffness_tensor(infinitesimal_strain)
        
        form_dV = + fdrk.inner(fdrk.grad(test_velocity), cauchy_stress)*fdrk.dx 
        return form_dV


    def mass_form(self, test, trial):
        return fdrk.inner(test, self.density*trial)*fdrk.dx


    def leapfrog(self, save_vars=False):
        """
        Solve using leapfrog/Verlet method
        Two version 
            - q at half-integers (the one introduced in the paper)
            - q at integers 
        Here we do at integers
        """
        dict_essential = self.get_essential_bcs()
        
        displacement_bc_data = dict_essential["displacement"]
        bcs_displacement = [fdrk.DirichletBC(self.space_displacement, item[1], item[0]) \
                        for item in displacement_bc_data.items()]

        velocity_bc_data = dict_essential["velocity"]
        bcs_velocity = [fdrk.DirichletBC(self.space_velocity, item[1], item[0]) \
                        for item in velocity_bc_data.items()]


        disp_old = fdrk.Function(self.space_displacement)
        vel_old = fdrk.Function(self.space_velocity)

        dict_init_conditions_disp = self.get_initial_conditions()

        disp_0 = dict_init_conditions_disp["displacement"]   
        disp_old.interpolate(disp_0)

        vel_0 = dict_init_conditions_disp["velocity"]   
        vel_old.interpolate(vel_0)

        disp_new = fdrk.Function(self.space_displacement)
        vel_new = fdrk.Function(self.space_velocity)

        test_vel = fdrk.TestFunction(self.space_velocity)
        trial_vel = fdrk.TrialFunction(self.space_velocity)

        disp_half = fdrk.Function(self.space_displacement)
        disp_half.assign(disp_old + 0.5*self.dt*vel_old)

        disp_new_half = disp_half + self.dt*vel_new

        dV_disp = self.weak_grad_potential(test_vel, disp_half)
        # dV_disp = self.weak_grad_potential_linear(test_vel, disp_half)

        mass_vel = self.mass_form(test_vel, trial_vel)
        rhs_vel  = self.mass_form(test_vel, vel_old) - self.dt * dV_disp

        problem_vel = fdrk.LinearVariationalProblem(mass_vel, rhs_vel, vel_new, bcs=bcs_velocity)
        solver_vel = fdrk.LinearVariationalSolver(problem_vel)

        if save_vars:   
            disp_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            vel_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            
            disp_array[0] = disp_old.dat.data_ro[:]
            vel_array[0] = vel_old.dat.data_ro[:]


        # energy_vec = np.zeros(len(self.t_vec_output)-1)
        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(disp_old, vel_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):
            solver_vel.solve()
            
            disp_new.assign(0.5*(disp_half + disp_new_half))      
             
            if (ii+1)%self.output_frequency==0:
                # energy_vec[kk] = fdrk.assemble(self.kinetic_energy(vel_new) \
                #                 + self.deformation_energy_leapfrog(disp_half, disp_new_half))
                kk += 1
                # actural_time = (ii+1)*float(self.dt)
                # assert np.isclose(actural_time, self.t_vec_output[kk])

                energy_vec[kk] = fdrk.assemble(self.hamiltonian(disp_new, vel_new))
                if save_vars: 
                    disp_array[kk] = disp_new.dat.data_ro[:]
                    vel_array[kk] = vel_new.dat.data_ro[:]

            disp_half.assign(disp_new_half)
            vel_old.assign(vel_new)

        if not save_vars:
            disp_array = disp_new.dat.data_ro[:]
            vel_array = vel_new.dat.data_ro[:]

        dict_results = {"displacement": disp_array, 
                        "velocity": vel_array, 
                        "energy": energy_vec}
        
        return dict_results
    

    def implicit_method(self, save_vars=False, type="implicit midpoint"):
        """
        Solve using leapfrog/Verlet method
        Two version 
            - q at half-integers (the one introduced in the paper)
            - q at integers 
        Here we do at integers
        """
        # bc_hor_disp = fdrk.DirichletBC(self.mixed_space_implicit.sub(0), fdrk.Constant(0), "on_boundary")        
        # bc_hor_vel = fdrk.DirichletBC(self.mixed_space_implicit.sub(2), fdrk.Constant(0), "on_boundary")

        bc_ver_disp = fdrk.DirichletBC(self.mixed_space_implicit.sub(1), fdrk.Constant(0), "on_boundary")
        bc_ver_vel = fdrk.DirichletBC(self.mixed_space_implicit.sub(3), fdrk.Constant(0), "on_boundary")

        # bcs = [bc_hor_disp, bc_ver_disp, bc_hor_vel, bc_ver_vel]
        bcs = [bc_ver_disp, bc_ver_vel]

        test_hor_disp, test_ver_disp, test_hor_vel, test_ver_vel = fdrk.TestFunctions(self.mixed_space_implicit)

        states_old = fdrk.Function(self.mixed_space_implicit) 
        hor_disp_old, ver_disp_old, hor_vel_old, ver_vel_old = states_old.subfunctions

        dict_init_conditions_disp = self.get_initial_conditions()

        hor_disp_0 = dict_init_conditions_disp["horizontal displacement"]
        hor_disp_old.interpolate(hor_disp_0)

        ver_disp_0 = dict_init_conditions_disp["vertical displacement"]
        ver_disp_old.assign(fdrk.project(ver_disp_0, self.space_ver_disp, \
                    fdrk.DirichletBC(self.space_ver_disp, fdrk.Constant(0), "on_boundary")))

        states_new = fdrk.Function(self.mixed_space_implicit)
        hor_disp_new, ver_disp_new, hor_vel_new, ver_vel_new = fdrk.split(states_new)

        states_new.sub(0).assign(hor_disp_old)
        states_new.sub(1).assign(ver_disp_old)

        hor_vel_midpoint = 0.5*(hor_vel_old + hor_vel_new)
        ver_vel_midpoint = 0.5*(ver_vel_old + ver_vel_new)

        ver_disp_midpoint = 0.5*(ver_disp_old + ver_disp_new)

        if type == "implicit midpoint":
            hor_disp_midpoint = 0.5*(hor_disp_old + hor_disp_new)

            dV_hor_disp, dV_ver_disp = self.weak_grad_potential(test_hor_vel, test_ver_vel, \
                                                                hor_disp_midpoint, ver_disp_midpoint)
        elif type == "discrete gradient":
            # To conserve the energy one needs the average axial stress and not the axial stress of the 
            # midpoint displacement (see Simo paper)
            axial_stress_old = self.axial_stress(hor_disp_old, ver_disp_old)
            axial_stress_new = self.axial_stress(hor_disp_new, ver_disp_new)
            average_axial_stress = 0.5*(axial_stress_old + axial_stress_new)

            bending_stress_midpoint = self.bending_stress(ver_disp_midpoint)

            dV_hor_disp, dV_ver_disp = self.weak_grad_potential_stress(test_hor_vel, test_ver_vel, \
                                    ver_disp_midpoint, average_axial_stress, bending_stress_midpoint)

        else:
            raise ValueError("Unknown type of implicit method")
        
        res_hor_disp = fdrk.inner(test_hor_disp, hor_disp_new - hor_disp_old - self.dt * hor_vel_midpoint)*fdrk.dx
        res_ver_disp = fdrk.inner(test_ver_disp, ver_disp_new - ver_disp_old - self.dt * ver_vel_midpoint)*fdrk.dx
        
        res_hor_vel = self.mass_form(test_hor_vel, hor_vel_new - hor_vel_old) \
                        + self.dt * dV_hor_disp
        res_ver_vel = self.mass_form(test_ver_vel, ver_vel_new - ver_vel_old) \
                        + self.dt * dV_ver_disp

        residual = res_hor_disp + res_ver_disp + res_hor_vel + res_ver_vel
        
        implicit_problem = fdrk.NonlinearVariationalProblem(residual, states_new, bcs=bcs)
        implicit_solver = fdrk.NonlinearVariationalSolver(implicit_problem)

        if save_vars:   
            hor_disp_array = np.zeros((self.n_output, self.n_dofs_hor))
            ver_disp_array = np.zeros((self.n_output, self.n_dofs_hor))

            hor_vel_array = np.zeros((self.n_output, self.n_dofs_hor))
            ver_vel_array = np.zeros((self.n_output, self.n_dofs_hor))
            
            hor_disp_array[0] = self.convert_function_to_array(hor_disp_old)
            ver_disp_array[0] = self.convert_function_to_array(ver_disp_old)

            hor_vel_array[0] = self.convert_function_to_array(hor_vel_old)
            ver_vel_array[0] = self.convert_function_to_array(ver_vel_old)

        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(hor_disp_old, ver_disp_old, hor_vel_old, ver_vel_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):

            implicit_solver.solve()

            states_old.assign(states_new)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                # actural_time = (ii+1)*float(self.dt)
                # assert np.isclose(actural_time, self.t_vec_output[kk])

                energy_vec[kk] = fdrk.assemble(self.hamiltonian(hor_disp_old, ver_disp_old, hor_vel_old, ver_vel_old))
                if save_vars: 
                    hor_disp_array[kk] = self.convert_function_to_array(hor_disp_old)
                    ver_disp_array[kk] = self.convert_function_to_array(ver_disp_old)

                    hor_vel_array[kk] = self.convert_function_to_array(hor_vel_old)
                    ver_vel_array[kk] = self.convert_function_to_array(ver_vel_old)

        if not save_vars:
            hor_disp_array = self.convert_function_to_array(hor_disp_old)
            ver_disp_array = self.convert_function_to_array(ver_disp_old)

            hor_vel_array = self.convert_function_to_array(hor_vel_old)
            ver_vel_array = self.convert_function_to_array(ver_vel_old)

        dict_results = {"horizontal displacement": hor_disp_array, 
                        "vertical displacement": ver_disp_array, 
                        "horizontal velocity": hor_vel_array, 
                        "vertical velocity": ver_vel_array, 
                        "energy": energy_vec}
        
        return dict_results
    

    def energy_form_linear_implicit(self, test_functions, trial_functions):

        test_hor_velocity, test_ver_velocity, test_axial_stress, test_bending_stress = test_functions
        trial_hor_velocity, trial_ver_velocity, trial_axial_stress, trial_bending_stress = trial_functions

        energy_density = fdrk.inner(test_hor_velocity, self.density*trial_hor_velocity) \
                       + fdrk.inner(test_ver_velocity, self.density*trial_ver_velocity) \
                       + fdrk.inner(test_axial_stress, self.axial_compliance*trial_axial_stress) \
                       + fdrk.inner(test_bending_stress, self.bending_compliance*trial_bending_stress)

        return energy_density*fdrk.dx
    

    def interconnection_form_linear_implicit(self, test_functions, trial_functions, vertical_disp):

        test_hor_velocity, test_ver_velocity, test_axial_stress, test_bending_stress = test_functions
        trial_hor_velocity, trial_ver_velocity, trial_axial_stress, trial_bending_stress = trial_functions

        D_transpose_hor, D_transpose_ver = self.weak_grad_potential_stress(test_hor_velocity, test_ver_velocity, \
                                    vertical_disp, trial_axial_stress, trial_bending_stress)
        
        D_hor, D_ver = self.weak_grad_potential_stress(trial_hor_velocity, trial_ver_velocity, \
                                    vertical_disp, test_axial_stress, test_bending_stress) 
        
        interconnection_form = - D_transpose_hor - D_transpose_ver + D_hor + D_ver 

        return interconnection_form


    def linear_implicit(self, save_vars=False):
        # bc_hor_vel = fdrk.DirichletBC(self.mixed_space_linear_implicit.sub(0), \
        #                             fdrk.Constant(0), "on_boundary")
        bc_ver_vel = fdrk.DirichletBC(self.mixed_space_linear_implicit.sub(1), \
                                      fdrk.Constant(0), "on_boundary")

        bcs = [bc_ver_vel]

        tuple_test_functions = fdrk.TestFunctions(self.mixed_space_linear_implicit)
        tuple_trial_functions = fdrk.TrialFunctions(self.mixed_space_linear_implicit)

        states_old = fdrk.Function(self.mixed_space_linear_implicit) 
        hor_vel_old, ver_vel_old, axial_stress_old, bending_stress_old = states_old.subfunctions 
        tuple_states_old = (hor_vel_old, ver_vel_old, axial_stress_old, bending_stress_old)

        hor_disp_old = fdrk.Function(self.space_hor_disp)
        ver_disp_old = fdrk.Function(self.space_ver_disp)

        dict_init_conditions_disp = self.get_initial_conditions()

        hor_disp_0 = dict_init_conditions_disp["horizontal displacement"]
        hor_disp_old.interpolate(hor_disp_0)

        ver_disp_0 = dict_init_conditions_disp["vertical displacement"]
        ver_disp_old.assign(fdrk.project(ver_disp_0, self.space_ver_disp, \
                    fdrk.DirichletBC(self.space_ver_disp, fdrk.Constant(0), "on_boundary")))
        
        states_old.sub(2).interpolate(self.axial_stress(hor_disp_old, ver_disp_old))
        states_old.sub(3).interpolate(self.bending_stress(ver_disp_old))


        states_new = fdrk.Function(self.mixed_space_linear_implicit)
        hor_vel_new, ver_vel_new, axial_stress_new, bending_stress_new = states_new.subfunctions
        tuple_states_new = (hor_vel_new, ver_vel_new, axial_stress_new, bending_stress_new)

        hor_disp_new = fdrk.Function(self.space_hor_disp)
        ver_disp_new = fdrk.Function(self.space_ver_disp)

        hor_disp_half = fdrk.Function(self.space_hor_disp)
        hor_disp_half.assign(hor_disp_old + 0.5*self.dt*hor_vel_old)

        ver_disp_half = fdrk.Function(self.space_ver_disp)
        ver_disp_half.assign(ver_disp_old + 0.5*self.dt*ver_vel_old)

        hor_disp_new_half = hor_disp_half + self.dt*hor_vel_new
        ver_disp_new_half = ver_disp_half + self.dt*ver_vel_new

        bilinear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_trial_functions) \
            - self.dt/2*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_trial_functions, \
                                                                ver_disp_half)


        linear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_states_old) \
            + self.dt/2*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_states_old, \
                                                                ver_disp_half)

        linear_implicit_problem = fdrk.LinearVariationalProblem(bilinear_form, \
                                                                linear_form, \
                                                                states_new, \
                                                                bcs=bcs)
        
        linear_implicit_solver = fdrk.LinearVariationalSolver(linear_implicit_problem)

        if save_vars:   
            hor_disp_array = np.zeros((self.n_output, self.n_dofs_hor))
            ver_disp_array = np.zeros((self.n_output, self.n_dofs_hor))

            hor_vel_array = np.zeros((self.n_output, self.n_dofs_hor))
            ver_vel_array = np.zeros((self.n_output, self.n_dofs_hor))
            
            hor_disp_array[0] = self.convert_function_to_array(hor_disp_old)
            ver_disp_array[0] = self.convert_function_to_array(ver_disp_old)

            hor_vel_array[0] = self.convert_function_to_array(hor_vel_old)
            ver_vel_array[0] = self.convert_function_to_array(ver_vel_old)


        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = 0.5* fdrk.assemble(self.energy_form_linear_implicit(tuple_states_old, \
                                                                            tuple_states_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):

            linear_implicit_solver.solve()

            hor_disp_new.assign(0.5*(hor_disp_half + hor_disp_new_half))      
            ver_disp_new.assign(0.5*(ver_disp_half + ver_disp_new_half))

            hor_disp_half.assign(hor_disp_new_half)
            ver_disp_half.assign(ver_disp_new_half)

            states_old.assign(states_new)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                # actural_time = (ii+1)*float(self.dt)
                # assert np.isclose(actural_time, self.t_vec_output[kk])
                energy_vec[kk] = 0.5*fdrk.assemble(self.energy_form_linear_implicit(tuple_states_new, \
                                                                                    tuple_states_new))
                
                if save_vars: 
                    hor_disp_array[kk] = self.convert_function_to_array(hor_disp_new)
                    ver_disp_array[kk] = self.convert_function_to_array(ver_disp_new)

                    hor_vel_array[kk] = self.convert_function_to_array(hor_vel_new)
                    ver_vel_array[kk] = self.convert_function_to_array(ver_vel_new)

        if not save_vars:
            hor_disp_array = self.convert_function_to_array(hor_disp_new)
            ver_disp_array = self.convert_function_to_array(ver_disp_new)

            hor_vel_array = self.convert_function_to_array(hor_vel_new)
            ver_vel_array = self.convert_function_to_array(ver_vel_new)


        dict_results = {"horizontal displacement": hor_disp_array, 
                        "vertical displacement": ver_disp_array, 
                        "horizontal velocity": hor_vel_array, 
                        "vertical velocity": ver_vel_array, 
                        "energy": energy_vec}
        
        return dict_results
    

    
    
    def linear_implicit_static_condensation(self, save_vars=False):
        tuple_test_functions = fdrk.TestFunctions(self.mixed_space_linear_implicit)
        tuple_trial_functions = fdrk.TrialFunctions(self.mixed_space_linear_implicit)

        states_old = fdrk.Function(self.mixed_space_linear_implicit) 
        hor_vel_old, ver_vel_old, axial_stress_old, bending_stress_old = states_old.subfunctions 
        tuple_states_old = (hor_vel_old, ver_vel_old, axial_stress_old, bending_stress_old)

        hor_disp_old = fdrk.Function(self.space_hor_disp)
        ver_disp_old = fdrk.Function(self.space_ver_disp)

        dict_init_conditions_disp = self.get_initial_conditions()

        hor_disp_0 = dict_init_conditions_disp["horizontal displacement"]
        hor_disp_old.interpolate(hor_disp_0)

        ver_disp_0 = dict_init_conditions_disp["vertical displacement"]
        ver_disp_old.assign(fdrk.project(ver_disp_0, self.space_ver_disp, \
                    fdrk.DirichletBC(self.space_ver_disp, fdrk.Constant(0), "on_boundary")))
        
        states_old.sub(2).interpolate(self.axial_stress(hor_disp_old, ver_disp_old))
        states_old.sub(3).interpolate(self.bending_stress(ver_disp_old))


        states_new = fdrk.Function(self.mixed_space_linear_implicit)
        hor_vel_new, ver_vel_new, axial_stress_new, bending_stress_new = states_new.subfunctions
        tuple_states_new = (hor_vel_new, ver_vel_new, axial_stress_new, bending_stress_new)

        hor_disp_new = fdrk.Function(self.space_hor_disp)
        ver_disp_new = fdrk.Function(self.space_ver_disp)

        hor_disp_half = fdrk.Function(self.space_hor_disp)
        hor_disp_half.assign(hor_disp_old + 0.5*self.dt*hor_vel_old)

        ver_disp_half = fdrk.Function(self.space_ver_disp)
        ver_disp_half.assign(ver_disp_old + 0.5*self.dt*ver_vel_old)

        hor_disp_new_half = hor_disp_half + self.dt*hor_vel_new
        ver_disp_new_half = ver_disp_half + self.dt*ver_vel_new

        bilinear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_trial_functions) \
            - self.dt/2*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_trial_functions, \
                                                                ver_disp_half)


        linear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_states_old) \
            + self.dt/2*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_states_old, \
                                                                ver_disp_half)


        # Apply static condensation to solve for the two velocity fields
        A_tensor = fdrk.Tensor(bilinear_form)
        A_blocks = A_tensor.blocks

        b_tensor = fdrk.Tensor(linear_form)
        b_blocks = b_tensor.blocks

        A_blocks_vel = A_blocks[:2, :2]  - A_blocks[:2, 2:] * A_blocks[2:, 2:].inv * A_blocks[2:, :2]
        b_blocks_vel = b_blocks[:2]  - A_blocks[:2, 2:] * A_blocks[2:, 2:].inv * b_blocks[2:]

        space_velocity = self.space_hor_vel * self.space_ver_vel
        velocity_new = fdrk.Function(space_velocity)

        # # bc_hor_vel = fdrk.DirichletBC(space_velocity.sub(0), \
        # #                             fdrk.Constant(0), "on_boundary")
        bc_ver_vel = fdrk.DirichletBC(space_velocity.sub(1), \
                                      fdrk.Constant(0), "on_boundary")
        bcs = [bc_ver_vel]

        if save_vars:   
            hor_disp_array = np.zeros((self.n_output, self.n_dofs_hor))
            ver_disp_array = np.zeros((self.n_output, self.n_dofs_hor))

            hor_vel_array = np.zeros((self.n_output, self.n_dofs_hor))
            ver_vel_array = np.zeros((self.n_output, self.n_dofs_hor))
            
            hor_disp_array[0] = self.convert_function_to_array(hor_disp_old)
            ver_disp_array[0] = self.convert_function_to_array(ver_disp_old)

            hor_vel_array[0] = self.convert_function_to_array(hor_vel_old)
            ver_vel_array[0] = self.convert_function_to_array(ver_vel_old)

        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = 0.5* fdrk.assemble(self.energy_form_linear_implicit(tuple_states_old, \
                                                                            tuple_states_old))
        kk = 0            

        M_inv_axial_stress = fdrk.assemble(A_blocks[2, 2].inv).M.handle
        M_inv_bend_stress = fdrk.assemble(A_blocks[3, 3].inv).M.handle

        for ii in tqdm(range(self.n_steps)):

            # Solve for velocity block
            A_vel = fdrk.assemble(A_blocks_vel, bcs=bcs)
            b_vel = fdrk.assemble(b_blocks_vel)
            fdrk.solve(A_vel, velocity_new, b_vel)

            states_new.sub(0).assign(velocity_new.sub(0))
            states_new.sub(1).assign(velocity_new.sub(1))

            # Reconstruction of the new state vector
            local_velocity = fdrk.AssembledVector(velocity_new)

            b_axial_stress= fdrk.assemble(b_blocks[2] - A_blocks[2, :2] * local_velocity)

            with b_axial_stress.dat.vec_ro as b_axial_stress_petsc:
                with axial_stress_new.dat.vec_wo as axial_stress_new_vec:
                    M_inv_axial_stress.mult(b_axial_stress_petsc, axial_stress_new_vec)
            
            b_bend_stress= fdrk.assemble(b_blocks[3] - A_blocks[3, :2] * local_velocity)

            with b_bend_stress.dat.vec_ro as b_bend_stress_petsc:
                with bending_stress_new.dat.vec_wo as bending_stress_new_vec:
                    M_inv_bend_stress.mult(b_bend_stress_petsc, bending_stress_new_vec)

            hor_disp_new.assign(0.5*(hor_disp_half + hor_disp_new_half))      
            ver_disp_new.assign(0.5*(ver_disp_half + ver_disp_new_half))

            hor_disp_half.assign(hor_disp_new_half)
            ver_disp_half.assign(ver_disp_new_half)

            states_old.assign(states_new)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                # actural_time = (ii+1)*float(self.dt)
                # assert np.isclose(actural_time, self.t_vec_output[kk])
                energy_vec[kk] = 0.5*fdrk.assemble(self.energy_form_linear_implicit(tuple_states_new, \
                                                                                    tuple_states_new))
                
                if save_vars: 
                    hor_disp_array[kk] = self.convert_function_to_array(hor_disp_new)
                    ver_disp_array[kk] = self.convert_function_to_array(ver_disp_new)

                    hor_vel_array[kk] = self.convert_function_to_array(hor_vel_new)
                    ver_vel_array[kk] = self.convert_function_to_array(ver_vel_new)

        if not save_vars:
            hor_disp_array = self.convert_function_to_array(hor_disp_new)
            ver_disp_array = self.convert_function_to_array(ver_disp_new)

            hor_vel_array = self.convert_function_to_array(hor_vel_new)
            ver_vel_array = self.convert_function_to_array(ver_vel_new)


        dict_results = {"horizontal displacement": hor_disp_array, 
                        "vertical displacement": ver_disp_array, 
                        "horizontal velocity": hor_vel_array, 
                        "vertical velocity": ver_vel_array, 
                        "energy": energy_vec}
        
        return dict_results