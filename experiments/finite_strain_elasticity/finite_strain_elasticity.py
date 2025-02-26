import firedrake as fdrk
import numpy as np
import os
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
        # self.space_velocity = fdrk.VectorFunctionSpace(self.domain, "CG", 1)
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
    

    def weak_grad_potential_linear(self, test_velocity, displacement):
        infinitesimal_strain = self.infinitesimal_strain(displacement)
        cauchy_stress = self.stiffness_tensor(infinitesimal_strain)
        
        form_dV = + fdrk.inner(fdrk.grad(test_velocity), cauchy_stress)*fdrk.dx 
        return form_dV
    

    def weak_grad_potential_stress(self, test_velocity, displacement, second_piola_stress_tensor):
        def_gradient = self.deformation_gradient(displacement)
        form_dV = fdrk.inner(fdrk.grad(test_velocity), fdrk.dot(def_gradient, second_piola_stress_tensor))*fdrk.dx 

        return form_dV


    def weak_grad_potential_stress_linear(self, test_velocity, second_piola_stress_tensor):
        form_dV = fdrk.inner(fdrk.grad(test_velocity), second_piola_stress_tensor)*fdrk.dx 

        return form_dV


    def mass_form(self, test, trial):
        return fdrk.inner(test, self.density*trial)*fdrk.dx


    def leapfrog(self, save_vars=False, paraview_directory=""):
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

        disp_new = fdrk.Function(self.space_displacement, name="disp_new")
        vel_new = fdrk.Function(self.space_velocity, name="vel_new")

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

        if paraview_directory != "":
            if not os.path.exists(paraview_directory):
                os.makedirs(paraview_directory)
            method_dir = paraview_directory + "/leapfrog/"
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            paraview_file = fdrk.File(method_dir + "/solution.pvd")

            disp_new.assign(disp_old)
            vel_new.assign(vel_old)

            paraview_file.write(disp_new, vel_new, time=0.0)


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
                actual_time = (ii+1)*float(self.dt)
                # assert np.isclose(actual_time, self.t_vec_output[kk])

                energy_vec[kk] = fdrk.assemble(self.hamiltonian(disp_new, vel_new))
                if save_vars: 
                    disp_array[kk] = disp_new.dat.data_ro[:]
                    vel_array[kk] = vel_new.dat.data_ro[:]

                if paraview_directory!="":
                    paraview_file.write(disp_new, vel_new, time=actual_time)

            disp_half.assign(disp_new_half)
            vel_old.assign(vel_new)

        if not save_vars:
            disp_array = disp_new.dat.data_ro[:]
            vel_array = vel_new.dat.data_ro[:]

        dict_results = {"displacement": disp_array, 
                        "velocity": vel_array, 
                        "energy": energy_vec}
        
        return dict_results
    

    def implicit_method(self, method="implicit midpoint", save_vars=False, paraview_directory=""):
        """
        Solve using leapfrog/Verlet method
        Two version 
            - q at half-integers (the one introduced in the paper)
            - q at integers 
        Here we do at integers
        """
        dict_essential = self.get_essential_bcs()
        
        displacement_bc_data = dict_essential["displacement"]
        bcs_displacement = [fdrk.DirichletBC(self.mixed_space_implicit.sub(0), item[1], item[0]) \
                        for item in displacement_bc_data.items()]

        velocity_bc_data = dict_essential["velocity"]
        bcs_velocity = [fdrk.DirichletBC(self.mixed_space_implicit.sub(1), item[1], item[0]) \
                        for item in velocity_bc_data.items()]
        
        bcs = bcs_displacement + bcs_velocity

        test_disp, test_vel = fdrk.TestFunctions(self.mixed_space_implicit)

        states_old = fdrk.Function(self.mixed_space_implicit, name="states_old") 
        disp_old, vel_old = states_old.subfunctions

        dict_init_conditions_disp = self.get_initial_conditions()

        disp_0 = dict_init_conditions_disp["displacement"]
        disp_old.interpolate(disp_0)

        vel_0 = dict_init_conditions_disp["velocity"]
        vel_old.interpolate(vel_0)

        states_new = fdrk.Function(self.mixed_space_implicit)
        disp_new, vel_new = fdrk.split(states_new)

        states_new.sub(0).assign(disp_old)
        states_new.sub(1).assign(vel_old)

        vel_midpoint = 0.5*(vel_old + vel_new)
        disp_midpoint = 0.5*(disp_old + disp_new)

        if method == "implicit midpoint":
            dV_disp = self.weak_grad_potential(test_vel, disp_midpoint)
        elif method == "discrete gradient":
            # To conserve the energy one needs the average axial stress and not the axial stress of the 
            # midpoint displacement (see Simo paper)
            stress_old = self.second_piola_stress(disp_old)
            stress_new = self.second_piola_stress(disp_new)
            average_stress = 0.5*(stress_old + stress_new)

            dV_disp = self.weak_grad_potential_stress(test_vel, disp_midpoint, average_stress)

        else:
            raise ValueError("Unknown type of implicit method")
        
        res_disp = fdrk.inner(test_disp, disp_new - disp_old - self.dt * vel_midpoint)*fdrk.dx
        res_vel = self.mass_form(test_vel, vel_new - vel_old) + self.dt * dV_disp
        
        residual = res_disp + res_vel
        
        implicit_problem = fdrk.NonlinearVariationalProblem(residual, states_new, bcs=bcs)
        implicit_solver = fdrk.NonlinearVariationalSolver(implicit_problem)

        if save_vars:   
            disp_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            vel_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            
            disp_array[0] = disp_old.dat.data_ro[:]
            vel_array[0] = vel_old.dat.data_ro[:]

        if paraview_directory != "":
            if not os.path.exists(paraview_directory):
                os.makedirs(paraview_directory)
            method_dir = paraview_directory + method
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            paraview_file = fdrk.File(method_dir + "/solution.pvd")

            paraview_file.write(disp_old, vel_old, time=0.0)

        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(disp_old, vel_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):

            implicit_solver.solve()
            states_old.assign(states_new)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                actual_time = (ii+1)*float(self.dt)
                # assert np.isclose(actural_time, self.t_vec_output[kk])
                energy_vec[kk] = fdrk.assemble(self.hamiltonian(disp_old, vel_old))
                if save_vars: 
                    disp_array[kk] = disp_old.dat.data_ro[:]
                    vel_array[kk] = vel_old.dat.data_ro[:]

                if paraview_directory!="":
                    paraview_file.write(disp_old, vel_old, time=actual_time)

        if not save_vars:
            disp_array = disp_old.dat.data_ro[:]
            vel_array = vel_old.dat.data_ro[:]

        dict_results = {"displacement": disp_array, 
                        "velocity": vel_array, 
                        "energy": energy_vec}
        
        return dict_results
    

    def energy_form_linear_implicit(self, test_functions, functions):

        test_velocity, test_stress = test_functions
        velocity, stress = functions

        energy_density = fdrk.inner(test_velocity, self.density*velocity) \
                       + fdrk.inner(test_stress, self.compliance_tensor(stress))

        return energy_density*fdrk.dx
    

    def interconnection_form_linear_implicit(self, test_functions, functions, displacement):

        test_velocity, test_stress = test_functions
        velocity, stress = functions

        # # This does not work with caching and static condensation
        D_transpose = self.weak_grad_potential_stress(test_velocity, displacement, stress)
        D = self.weak_grad_potential_stress(velocity, displacement, test_stress) 
        interconnection_form = - D_transpose  + D

        # # # The linear version works with caching and static condensation
        # D_transpose = self.weak_grad_potential_stress_linear(test_velocity, stress)
        # D = self.weak_grad_potential_stress_linear(velocity, test_stress) 
        # interconnection_form = - D_transpose  + D

        return interconnection_form


    def linear_implicit(self, save_vars=False, paraview_directory=""):
        dict_essential = self.get_essential_bcs()
        
        # displacement_bc_data = dict_essential["displacement"]
        # bcs_displacement = [fdrk.DirichletBC(self.space_displacement, item[1], item[0]) \
        #                 for item in displacement_bc_data.items()]

        velocity_bc_data = dict_essential["velocity"]
        bcs_velocity = [fdrk.DirichletBC(self.mixed_space_linear_implicit.sub(0), item[1], item[0]) \
                        for item in velocity_bc_data.items()]
        
        bcs = bcs_velocity
        
        tuple_test_functions = fdrk.TestFunctions(self.mixed_space_linear_implicit)
        tuple_trial_functions = fdrk.TrialFunctions(self.mixed_space_linear_implicit)

        states_old = fdrk.Function(self.mixed_space_linear_implicit, name="states_old") 
        vel_old, stress_old = states_old.subfunctions 
        tuple_states_old = (vel_old, stress_old)

        disp_old = fdrk.Function(self.space_displacement, name="disp_old")

        dict_init_conditions_disp = self.get_initial_conditions()
        disp_0 = dict_init_conditions_disp["displacement"]
        disp_old.interpolate(disp_0)

        vel_0 = dict_init_conditions_disp["velocity"]
        vel_old.interpolate(vel_0)

        stress_old.interpolate(self.second_piola_stress(disp_old))

        states_new = fdrk.Function(self.mixed_space_linear_implicit, name="states_new")
        vel_new, stress_new = states_new.subfunctions
        tuple_states_new = (vel_new, stress_new)

        disp_new = fdrk.Function(self.space_displacement, name="disp_new")

        disp_half = fdrk.Function(self.space_displacement)
        disp_half.assign(disp_old + 0.5*self.dt*vel_old)
        disp_new_half = disp_half + self.dt*vel_new

        bilinear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_trial_functions) \
            - self.dt/2*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_trial_functions, \
                                                                disp_half)


        linear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_states_old) \
            + self.dt/2*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_states_old, \
                                                                disp_half)

        linear_implicit_problem = fdrk.LinearVariationalProblem(bilinear_form, \
                                                                linear_form, \
                                                                states_new, \
                                                                bcs=bcs)
        
        linear_implicit_solver = fdrk.LinearVariationalSolver(linear_implicit_problem)

        if save_vars:   
            disp_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            vel_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            
            disp_array[0] = disp_old.dat.data_ro[:]
            vel_array[0] = vel_old.dat.data_ro[:]

        if paraview_directory != "":
            if not os.path.exists(paraview_directory):
                os.makedirs(paraview_directory)
            method_dir = paraview_directory + "/linear_implicit/"
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            paraview_file = fdrk.File(method_dir + "/solution.pvd")

            disp_new.assign(disp_old)
            vel_new.assign(vel_old)
            paraview_file.write(disp_new, vel_new, time=0.0)


        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = 0.5* fdrk.assemble(self.energy_form_linear_implicit(tuple_states_old, \
                                                                            tuple_states_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):

            linear_implicit_solver.solve()

            disp_new.assign(0.5*(disp_half + disp_new_half))
            disp_half.assign(disp_new_half)

            states_old.assign(states_new)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                actual_time = (ii+1)*float(self.dt)
                # assert np.isclose(actural_time, self.t_vec_output[kk])
                energy_vec[kk] = 0.5*fdrk.assemble(self.energy_form_linear_implicit(tuple_states_new, \
                                                                                    tuple_states_new))
                if save_vars: 
                    disp_array[kk] = disp_new.dat.data_ro[:]
                    vel_array[kk] = vel_new.dat.data_ro[:]

                if paraview_directory!="":
                    paraview_file.write(disp_new, vel_new, time=actual_time)

        if not save_vars:
            disp_array = disp_new.dat.data_ro[:]
            vel_array = vel_new.dat.data_ro[:]

        dict_results = {"displacement": disp_array, 
                        "velocity": vel_array, 
                        "energy": energy_vec}
        
        return dict_results
    

    def linear_implicit_static_condensation(self, save_vars=False, paraview_directory=""):
        
        tuple_test_functions = fdrk.TestFunctions(self.mixed_space_linear_implicit)
        tuple_trial_functions = fdrk.TrialFunctions(self.mixed_space_linear_implicit)

        states_old = fdrk.Function(self.mixed_space_linear_implicit, name="states_old") 
        tuple_states_old = states_old.subfunctions 
        vel_old, stress_old = tuple_states_old

        disp_old = fdrk.Function(self.space_displacement, name="disp_old")

        dict_init_conditions_disp = self.get_initial_conditions()
        disp_0 = dict_init_conditions_disp["displacement"]
        disp_old.interpolate(disp_0)

        vel_0 = dict_init_conditions_disp["velocity"]

        vel_old.interpolate(vel_0)
        stress_old.interpolate(self.second_piola_stress(disp_old))

        states_new = fdrk.Function(self.mixed_space_linear_implicit, name="states_new")
        tuple_states_new = states_new.subfunctions
        _, stress_new = tuple_states_new

        disp_new = fdrk.Function(self.space_displacement, name="disp_new")
        vel_new = fdrk.Function(self.space_velocity, name="vel_new")
        
        dict_essential = self.get_essential_bcs()
        velocity_bc_data = dict_essential["velocity"]
        bcs_velocity = [fdrk.DirichletBC(self.space_velocity, item[1], item[0]) \
                        for item in velocity_bc_data.items()]

        disp_half = fdrk.Function(self.space_displacement, name="disp_half")
        disp_half.assign(disp_old + 0.5*self.dt*vel_old)

        disp_new_half = disp_half + self.dt*vel_new

        bilinear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_trial_functions)  \
            - 0.5*self.dt*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_trial_functions, \
                                                                disp_half)

        linear_form = self.energy_form_linear_implicit(tuple_test_functions, tuple_states_old) \
            + 0.5*self.dt*self.interconnection_form_linear_implicit(tuple_test_functions, \
                                                                tuple_states_old, \
                                                                disp_half)

        # Apply static condensation to solve for velocity only

        A_tensor = fdrk.Tensor(bilinear_form) 
        A_blocks = A_tensor.blocks

        A_11_inv = A_blocks[1, 1].inv

        A_block_vel = A_blocks[0, 0] - A_blocks[0, 1] * A_11_inv * A_blocks[1, 0]

        b_vector = fdrk.Tensor(linear_form)
        b_blocks = b_vector.blocks

        b_block_vel = b_blocks[0] - A_blocks[0, 1] * A_11_inv * b_blocks[1]
        
        # # # Misterious bug. This is not working as expected        
        # linear_implicit_velocity_problem = fdrk.LinearVariationalProblem(A_block_vel, \
        #                                                                 b_block_vel, \
        #                                                                 vel_new, \
        #                                                                 bcs=bcs_velocity)
        
        # linear_implicit_velocity_solver = fdrk.LinearVariationalSolver(linear_implicit_velocity_problem)

        if save_vars:   
            disp_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            vel_array = np.zeros((self.n_output, self.n_dofs_disp, 3))
            
            disp_array[0] = disp_old.dat.data_ro[:]
            vel_array[0] = vel_old.dat.data_ro[:]

        if paraview_directory != "":
            if not os.path.exists(paraview_directory):
                os.makedirs(paraview_directory)
            method_dir = paraview_directory + "/linear_implicit/"
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
            paraview_file = fdrk.File(method_dir + "/solution.pvd")

            disp_new.assign(disp_old)
            vel_new.assign(vel_old)
            paraview_file.write(disp_new, vel_new, time=0.0)


        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = 0.5* fdrk.assemble(self.energy_form_linear_implicit(tuple_states_old, \
                                                                            tuple_states_old))
        kk = 0

        M_inv_stress = fdrk.assemble(A_11_inv).M.handle

        for ii in tqdm(range(self.n_steps)):

            # linear_implicit_velocity_solver.solve()

            # local_vel_new = fdrk.AssembledVector(vel_new)

            # stress_new = fdrk.assemble(A_11_inv * (b_blocks[1] \
            #                             - A_blocks[1, 0] * local_vel_new))
            
            # states_new.sub(0).assign(vel_new)
            # states_new.sub(1).vector().set_local(stress_new.vector().get_local())

            A_vel = fdrk.assemble(A_block_vel, bcs=bcs_velocity)
            b_vel = fdrk.assemble(b_block_vel)
            fdrk.solve(A_vel, vel_new, b_vel)

            states_new.sub(0).assign(vel_new)

            local_vel_new = fdrk.AssembledVector(vel_new)
            
            b_stress= fdrk.assemble(b_blocks[1] - A_blocks[1, 0] * local_vel_new)

            with b_stress.dat.vec_ro as b_stress_petsc:
                with stress_new.dat.vec_wo as stress_new_vec:
                    M_inv_stress.mult(b_stress_petsc, stress_new_vec)
            
            disp_new.assign(0.5*(disp_half + disp_new_half))
            disp_half.assign(disp_new_half)

            states_old.assign(states_new)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                actual_time = (ii+1)*float(self.dt)
                # assert np.isclose(actural_time, self.t_vec_output[kk])
                energy_vec[kk] = 0.5*fdrk.assemble(self.energy_form_linear_implicit(tuple_states_new, \
                                                                                    tuple_states_new))
                if save_vars: 
                    disp_array[kk] = disp_new.dat.data_ro[:]
                    vel_array[kk] = vel_new.dat.data_ro[:]

                if paraview_directory!="":
                    paraview_file.write(disp_new, vel_new, time=actual_time)

        if not save_vars:
            disp_array = disp_new.dat.data_ro[:]
            vel_array = vel_new.dat.data_ro[:]

        dict_results = {"displacement": disp_array, 
                        "velocity": vel_array, 
                        "energy": energy_vec}
        
        return dict_results
    

    