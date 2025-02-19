import firedrake as fdrk
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class VonKarmanBeam:
    def __init__(self, **kwargs): 
        parameters = {
        "rho":1, "E": 1, "A": 1, "I":1, "L":1,
        "q0_hor":1, "q0_ver":1, "time_step": 0.01, "n_elem": 10,  
        "n_output": 100, "t_span": np.array([0, 1]),
        }

        for key, value in kwargs.items():
            if key in parameters:
                # print(f"Paramter {key} modified from default value {parameters[key]}")
                parameters[key] = value

        # Physical parameters
        rho = parameters["rho"]
        E = parameters["E"]
        A = parameters["A"]
        I = parameters["I"]

        self.density = rho*A
        self.bending_stiffness = E*I
        self.axial_stiffness = E*A
        self.bending_compliance = 1/self.bending_stiffness
        self.axial_compliance = 1/self.axial_stiffness
        self.length = parameters["L"]

        # Initial conditions
        self.q0_hor = parameters["q0_hor"]
        self.q0_ver = parameters["q0_ver"]

        # Mesh and space parameters
        self.n_elem = parameters["n_elem"]
        self.domain = fdrk.IntervalMesh(self.n_elem, self.length)
        self.x_coord = fdrk.SpatialCoordinate(self.domain)[0]
        x_data = self.domain.coordinates.dat.data_ro[:]
        x_sorted_idx = np.argsort(x_data)
        self.x_vec = x_data[x_sorted_idx]

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
        self.space_hor_disp = fdrk.FunctionSpace(self.domain, "CG", 1)
        self.space_hor_vel = self.space_hor_disp

        self.space_ver_disp = fdrk.FunctionSpace(self.domain, "Hermite", 3)
        self.space_ver_vel = self.space_ver_disp
        # When non linear effects are strong the degree for the axial variable needs
        # to be high
        self.space_axial_stress = fdrk.FunctionSpace(self.domain, "DG", 4)
        self.space_bending_stress = fdrk.FunctionSpace(self.domain, "DG", 1)

        self.mixed_space_implicit = self.space_hor_disp * self.space_ver_disp * self.space_hor_vel * self.space_ver_vel 
        self.mixed_space_linear_implicit = self.space_hor_vel * self.space_ver_vel * self.space_axial_stress * self.space_bending_stress


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
        hor_disp_exp = self.q0_hor*fdrk.cos(fdrk.pi*self.x_coord/self.length)
        
        ver_disp_exp = self.q0_ver*fdrk.sin(fdrk.pi*self.x_coord/self.length)
        return {"horizontal displacement": hor_disp_exp, "vertical displacement": ver_disp_exp}


    def axial_strain(self, hor_disp, ver_disp):
        try:
            eps_a = hor_disp.dx(0) + 1/2*(ver_disp.dx(0))**2
            return eps_a
        except ValueError:
            print("Invalid expression in axial strain. Don't use numbers.")
            return


    def bending_strain(self, ver_disp):
        try:
            kappa = ver_disp.dx(0).dx(0) 
        except ValueError:
            print("Invalid expression in bending strain. Don't use numbers.")
        return kappa
    
    
    def axial_stress(self, hor_disp, ver_disp):
        return self.axial_stiffness*self.axial_strain(hor_disp, ver_disp)
    

    def bending_stress(self, ver_disp):
        return self.bending_stiffness*self.bending_strain(ver_disp)
    

    def deformation_energy(self, hor_disp, ver_disp):
        potential_energy_density = 0.5*(self.axial_stiffness*self.axial_strain(hor_disp, ver_disp)**2 \
                                    + self.bending_stiffness*self.bending_strain(ver_disp)**2)
        return potential_energy_density*fdrk.dx


    def kinetic_energy(self, hor_vel, ver_vel):
        kinetic_energy_density = 0.5*self.density*(hor_vel**2 + ver_vel**2)
        return kinetic_energy_density*fdrk.dx


    def hamiltonian(self, hor_disp, ver_disp, hor_vel, ver_vel):
        return self.kinetic_energy(hor_vel, ver_vel) + self.deformation_energy(hor_disp, ver_disp)
    

    def deformation_energy_leapfrog(self, hor_disp_min, hor_disp_plus, ver_disp_min, ver_disp_plus):
        axial_energy_density = 0.5*self.axial_stiffness*fdrk.inner(hor_disp_min.dx(0), hor_disp_plus.dx(0))
        bending_energy_density = 0.5*self.bending_stiffness*fdrk.inner(ver_disp_min.dx(0).dx(0), ver_disp_plus.dx(0).dx(0))
        deformation_density = axial_energy_density + bending_energy_density

        return deformation_density*fdrk.dx 
        

    def weak_grad_potential(self, test_hor_vel, test_ver_vel, hor_disp, ver_disp):
        axial_stress = self.axial_stress(hor_disp, ver_disp)
        bending_stress = self.bending_stress(ver_disp)

        form_dV_hor_disp = + fdrk.inner(test_hor_vel.dx(0), axial_stress)*fdrk.dx 
        
        form_dV_ver_disp = + fdrk.inner(test_ver_vel.dx(0).dx(0), bending_stress)*fdrk.dx \
                      + fdrk.inner(test_ver_vel.dx(0), axial_stress*ver_disp.dx(0))*fdrk.dx
        
        return form_dV_hor_disp, form_dV_ver_disp
    
    def weak_grad_potential_stress(self, test_hor_vel, test_ver_vel, ver_disp, \
                                    axial_stress, bending_stress):

        form_dV_hor_disp = + fdrk.inner(test_hor_vel.dx(0), axial_stress)*fdrk.dx 
        
        form_dV_ver_disp = + fdrk.inner(test_ver_vel.dx(0).dx(0), bending_stress)*fdrk.dx \
                      + fdrk.inner(test_ver_vel.dx(0), axial_stress*ver_disp.dx(0))*fdrk.dx
        

        return form_dV_hor_disp, form_dV_ver_disp

    def weak_grad_potential_linear(self, test_hor_vel, test_ver_vel, \
                                            hor_disp, ver_disp):
        N_xx = self.axial_stiffness*hor_disp.dx(0)
        bend_mom = self.bending_stress(ver_disp)

        form_dV_hor_disp = + fdrk.inner(test_hor_vel.dx(0), N_xx)*fdrk.dx 
        
        form_dV_ver_disp = + fdrk.inner(test_ver_vel.dx(0).dx(0), bend_mom)*fdrk.dx 
        
        return form_dV_hor_disp, form_dV_ver_disp
    

    def mass_form(self, test, trial):
        return fdrk.inner(test, self.density*trial)*fdrk.dx


    def convert_functions_to_array(self, list_functions):
        """
        Convert a list of function to an array of size n_t * n_dofs
        where n_t is len(list_functions) (the number of simulation instants)
        """

        n_times = len(list_functions)

        f_0 = list_functions[0] 
        V = f_0.function_space()
        finite_element = V.ufl_element()

        is_hermite = 'Hermite' in finite_element.family()
        
        if is_hermite:
            n_dofs = int(V.dim()/2)
        else:
            n_dofs = V.dim()

        y_matrix = np.zeros((n_times, n_dofs))


        for count, function in enumerate(list_functions):
            if is_hermite:
                
                y_matrix[count, :] =  function.dat.data_ro[::2]
            else:
                y_matrix[count, :] =  function.dat.data_ro[:]


        return y_matrix
    

    def leapfrog(self, save_vars=False):
        """
        Solve using leapfrog/Verlet method
        Two version 
            - q at half-integers (the one introduced in the paper)
            - q at integers 
        Here we do at integers
        """
        # bc_hor_vel = fdrk.DirichletBC(self.space_hor_vel, fdrk.Constant(0), "on_boundary")
        bc_hor_vel = []
        bc_ver_disp = fdrk.DirichletBC(self.space_ver_disp, fdrk.Constant(0), "on_boundary")
        bc_ver_vel = fdrk.DirichletBC(self.space_ver_vel, fdrk.Constant(0), "on_boundary")

        hor_disp_old = fdrk.Function(self.space_hor_disp)
        ver_disp_old = fdrk.Function(self.space_ver_disp)

        hor_vel_old = fdrk.Function(self.space_hor_vel)
        ver_vel_old = fdrk.Function(self.space_ver_vel)

        dict_init_conditions_disp = self.get_initial_conditions()

        hor_disp_0 = dict_init_conditions_disp["horizontal displacement"]   
        hor_disp_old.interpolate(hor_disp_0)

        ver_disp_0 = dict_init_conditions_disp["vertical displacement"]
        ver_disp_old.assign(fdrk.project(ver_disp_0, self.space_ver_disp, bcs = bc_ver_disp))

        hor_disp_new = fdrk.Function(self.space_hor_disp)
        ver_disp_new = fdrk.Function(self.space_ver_disp)

        hor_vel_new = fdrk.Function(self.space_hor_vel)
        ver_vel_new = fdrk.Function(self.space_ver_vel)

        test_hor_vel = fdrk.TestFunction(self.space_hor_vel)
        test_ver_vel = fdrk.TestFunction(self.space_ver_vel)
        trial_hor_vel = fdrk.TrialFunction(self.space_hor_vel)
        trial_ver_vel = fdrk.TrialFunction(self.space_ver_vel)

        hor_disp_half = fdrk.Function(self.space_hor_disp)
        hor_disp_half.assign(hor_disp_old + 0.5*self.dt*hor_vel_old)

        ver_disp_half = fdrk.Function(self.space_ver_disp)
        ver_disp_half.assign(ver_disp_old + 0.5*self.dt*ver_vel_old)

        hor_disp_new_half = hor_disp_half + self.dt*hor_vel_new
        ver_disp_new_half = ver_disp_half + self.dt*ver_vel_new

        dV_hor_disp, dV_ver_disp = self.weak_grad_potential(test_hor_vel, test_ver_vel, hor_disp_half, ver_disp_half)
        # dV_hor_disp, dV_ver_disp = self.weak_grad_potential_linear(test_hor_vel, test_ver_vel, hor_disp_half, ver_disp_half)

        mass_hor_vel = self.mass_form(test_hor_vel, trial_hor_vel)
        rhs_hor_vel  = self.mass_form(test_hor_vel, hor_vel_old) - self.dt * dV_hor_disp

        problem_hor_vel = fdrk.LinearVariationalProblem(mass_hor_vel, rhs_hor_vel, hor_vel_new, bcs=bc_hor_vel)
        solver_hor_vel = fdrk.LinearVariationalSolver(problem_hor_vel)

        mass_ver_vel = self.mass_form(test_ver_vel, trial_ver_vel)
        rhs_ver_vel = self.mass_form(test_ver_vel, ver_vel_old) - self.dt * dV_ver_disp

        problem_ver_vel = fdrk.LinearVariationalProblem(mass_ver_vel, rhs_ver_vel, ver_vel_new, bcs=bc_ver_vel)
        
        solver_ver_vel = fdrk.LinearVariationalSolver(problem_ver_vel)

        hor_disp_list = []
        ver_disp_list = []
        hor_vel_list = []
        ver_vel_list = []

        if save_vars:   
            hor_disp_list.append(hor_disp_old.copy(deepcopy=True))
            ver_disp_list.append(ver_disp_old.copy(deepcopy=True))
            hor_vel_list.append(hor_vel_old.copy(deepcopy=True))
            ver_vel_list.append(ver_vel_old.copy(deepcopy=True))

        # energy_vec = np.zeros(len(self.t_vec_output)-1)
        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(hor_disp_old, ver_disp_old, hor_vel_old, ver_vel_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):
            actural_time = (ii+1)*float(self.dt)
            solver_hor_vel.solve()
            solver_ver_vel.solve()
            
            hor_disp_new.assign(0.5*(hor_disp_half + hor_disp_new_half))      
            ver_disp_new.assign(0.5*(ver_disp_half + ver_disp_new_half))

             
            if (ii+1)%self.output_frequency==0:
                # energy_vec[kk] = fdrk.assemble(self.kinetic_energy(hor_vel_new, ver_vel_new) \
                #                 + self.deformation_energy_leapfrog(hor_disp_half, hor_disp_new_half, ver_disp_half, ver_disp_new_half))
                kk += 1
                assert np.isclose(actural_time, self.t_vec_output[kk])

                energy_vec[kk] = fdrk.assemble(self.hamiltonian(hor_disp_new, ver_disp_new, hor_vel_new, ver_vel_new))
                if save_vars: 
                    hor_disp_list.append(hor_disp_new.copy(deepcopy=True))
                    ver_disp_list.append(ver_disp_new.copy(deepcopy=True))
                    hor_vel_list.append(hor_vel_new.copy(deepcopy=True))
                    ver_vel_list.append(ver_vel_new.copy(deepcopy=True))

            hor_disp_half.assign(hor_disp_new_half)
            ver_disp_half.assign(ver_disp_new_half)

            hor_vel_old.assign(hor_vel_new)
            ver_vel_old.assign(ver_vel_new)

        if not save_vars:
            hor_disp_list.append(hor_disp_old.copy(deepcopy=True))
            ver_disp_list.append(ver_disp_old.copy(deepcopy=True))
            hor_vel_list.append(hor_vel_old.copy(deepcopy=True))
            ver_vel_list.append(ver_vel_old.copy(deepcopy=True))

        hor_disp_array = self.convert_functions_to_array(hor_disp_list)
        ver_disp_array = self.convert_functions_to_array(ver_disp_list)
        hor_vel_array = self.convert_functions_to_array(hor_vel_list)
        ver_vel_array = self.convert_functions_to_array(ver_vel_list)
           
        dict_results = {"horizontal displacement": hor_disp_array, 
                        "vertical displacement": ver_disp_array, 
                        "horizontal velocity": hor_vel_array, 
                        "vertical velocity": ver_vel_array, 
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

        hor_disp_list = []
        ver_disp_list = []
        hor_vel_list = []
        ver_vel_list = []

        if save_vars:   
            hor_disp_list.append(hor_disp_old.copy(deepcopy=True))
            ver_disp_list.append(ver_disp_old.copy(deepcopy=True))
            hor_vel_list.append(hor_vel_old.copy(deepcopy=True))
            ver_vel_list.append(ver_vel_old.copy(deepcopy=True))

        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(hor_disp_old, ver_disp_old, hor_vel_old, ver_vel_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):
            actural_time = (ii+1)*float(self.dt)

            implicit_solver.solve()

            states_old.assign(states_new)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                assert np.isclose(actural_time, self.t_vec_output[kk])

                energy_vec[kk] = fdrk.assemble(self.hamiltonian(hor_disp_old, ver_disp_old, hor_vel_old, ver_vel_old))
                if save_vars: 
                    hor_disp_list.append(hor_disp_old.copy(deepcopy=True))
                    ver_disp_list.append(ver_disp_old.copy(deepcopy=True))
                    hor_vel_list.append(hor_vel_old.copy(deepcopy=True))
                    ver_vel_list.append(ver_vel_old.copy(deepcopy=True))

        if not save_vars:
            hor_disp_list.append(hor_disp_old.copy(deepcopy=True))
            ver_disp_list.append(ver_disp_old.copy(deepcopy=True))
            hor_vel_list.append(hor_vel_old.copy(deepcopy=True))
            ver_vel_list.append(ver_vel_old.copy(deepcopy=True))


        hor_disp_array = self.convert_functions_to_array(hor_disp_list)
        ver_disp_array = self.convert_functions_to_array(ver_disp_list)
        hor_vel_array = self.convert_functions_to_array(hor_vel_list)
        ver_vel_array = self.convert_functions_to_array(ver_vel_list)
           
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

        hor_disp_list = []
        ver_disp_list = []
        hor_vel_list = []
        ver_vel_list = []

        if save_vars:   
            hor_disp_list.append(hor_disp_old.copy(deepcopy=True))
            ver_disp_list.append(ver_disp_old.copy(deepcopy=True))
            hor_vel_list.append(hor_vel_old.copy(deepcopy=True))
            ver_vel_list.append(ver_vel_old.copy(deepcopy=True))

        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = 0.5* fdrk.assemble(self.energy_form_linear_implicit(tuple_states_old, \
                                                                            tuple_states_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):
            actural_time = (ii+1)*float(self.dt)

            linear_implicit_solver.solve()

            hor_disp_new.assign(0.5*(hor_disp_half + hor_disp_new_half))      
            ver_disp_new.assign(0.5*(ver_disp_half + ver_disp_new_half))

            if (ii+1)%self.output_frequency==0:
                # energy_vec[kk] = fdrk.assemble(self.kinetic_energy(hor_vel_new, ver_vel_new) \
                #                 + self.deformation_energy_leapfrog(hor_disp_half, hor_disp_new_half, ver_disp_half, ver_disp_new_half))
                kk += 1
                assert np.isclose(actural_time, self.t_vec_output[kk])
                energy_vec[kk] = 0.5*fdrk.assemble(self.energy_form_linear_implicit(tuple_states_new, \
                                                                                    tuple_states_new))
                if save_vars: 
                    hor_disp_list.append(hor_disp_new.copy(deepcopy=True))
                    ver_disp_list.append(ver_disp_new.copy(deepcopy=True))
                    hor_vel_list.append(hor_vel_new.copy(deepcopy=True))
                    ver_vel_list.append(ver_vel_new.copy(deepcopy=True))

            hor_disp_half.assign(hor_disp_new_half)
            ver_disp_half.assign(ver_disp_new_half)

            states_old.assign(states_new)

        if not save_vars:
            hor_disp_list.append(hor_disp_old.copy(deepcopy=True))
            ver_disp_list.append(ver_disp_old.copy(deepcopy=True))
            hor_vel_list.append(hor_vel_old.copy(deepcopy=True))
            ver_vel_list.append(ver_vel_old.copy(deepcopy=True))
           
        hor_disp_array = self.convert_functions_to_array(hor_disp_list)
        ver_disp_array = self.convert_functions_to_array(ver_disp_list)
        hor_vel_array = self.convert_functions_to_array(hor_vel_list)
        ver_vel_array = self.convert_functions_to_array(ver_vel_list)
           
        dict_results = {"horizontal displacement": hor_disp_array, 
                        "vertical displacement": ver_disp_array, 
                        "horizontal velocity": hor_vel_array, 
                        "vertical velocity": ver_vel_array, 
                        "energy": energy_vec}
        
        return dict_results