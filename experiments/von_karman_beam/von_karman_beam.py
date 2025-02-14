import firedrake as fdrk
import numpy as np
from tqdm import tqdm
from src.discrete_gradient import discrete_gradient_firedrake_twofield

class VonKarmanBeam:
    def __init__(self, **kwargs): 
        parameters = {
        "rho":1, "E": 1, "A": 1, "I":1, "L":1,
        "q0":1, "time_step": 0.01, "n_elem": 10,  
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

        # Simulation parameters
        self.n_elem = parameters["n_elem"]
        self.domain = fdrk.IntervalMesh(self.n_elem, self.length)
        self.x_coord = fdrk.SpatialCoordinate(self.domain)[0]
        x_data = self.domain.coordinates.dat.data_ro[:]
        x_sorted_idx = np.argsort(x_data)
        self.x_vec = x_data[x_sorted_idx]

        self.q0 = parameters["q0"]
        time_step = parameters["time_step"]

        t_span = parameters["t_span"]
        simulation_time = t_span[1] - t_span[0]
        self.n_steps =np.round(simulation_time/time_step).astype(int)
        T_init = t_span[0]
        T_end = self.n_steps*time_step + T_init
        self.dt = fdrk.Constant(time_step)
        self.t_span = np.array([T_init, T_end])
        self.t_vec = np.linspace(T_init, T_end, self.n_steps+1)

        # Output parameters
        self.n_output = parameters["n_output"]
        n_sim_times = len(self.t_vec)
        if n_sim_times>self.n_output:
            self.output_frequency = int(n_sim_times/self.n_output)
        else:
            self.output_frequency = 1
        self.t_vec_output = self.t_vec[::self.output_frequency]

        # Finite elemnt spaces
        self.space_hor_displacement = fdrk.FunctionSpace(self.domain, "CG", 1)
        self.space_hor_velocity = self.space_hor_displacement

        self.space_ver_displacement = fdrk.FunctionSpace(self.domain, "Hermite", 3)
        self.space_ver_velocity = self.space_ver_displacement

        self.space_axial_stress = fdrk.FunctionSpace(self.domain, "DG", 0)
        self.space_bending_stress = fdrk.FunctionSpace(self.domain, "DG", 1)

        self.mixed_space_implicit = self.space_hor_displacement * self.space_ver_displacement * self.space_hor_velocity * self.space_ver_velocity 
        self.mixed_space_linear_implicit = self.space_hor_velocity * self.space_ver_velocity * self.space_axial_stress * self.space_bending_stress


    def get_initial_conditions_ver_displacement(self):
        ver_displacement_exp = self.q0*fdrk.sin(fdrk.pi*self.x_coord/self.length)
        return ver_displacement_exp


    def axial_strain(self, hor_displacement, ver_displacement):
        try:
            eps_a = hor_displacement.dx(0) + 1/2*(ver_displacement.dx(0))**2
            return eps_a
        except ValueError:
            print("Invalid expression in axial strain. Don't use numbers.")
            return

    def bending_strain(self, ver_displacement):
        try:
            kappa = ver_displacement.dx(0).dx(0) 
        except ValueError:
            print("Invalid expression in bending strain. Don't use numbers.")
        return kappa
    
    
    def axial_stress(self, hor_displacement, ver_displacement):
        return self.axial_stiffness*self.axial_strain(hor_displacement, ver_displacement)
    
    def bending_stress(self, ver_displacement):
        return self.bending_stiffness*self.bending_strain(ver_displacement)
    

    def deformation_energy(self, hor_displacement, ver_displacement):
        potential_energy_density = 0.5*(self.axial_stiffness*self.axial_strain(hor_displacement, ver_displacement)**2 \
                                    + self.bending_stiffness*self.bending_strain(ver_displacement)**2)
        return potential_energy_density*fdrk.dx

    def kinetic_energy(self, hor_velocity, ver_velocity):
        kinetic_energy_density = 0.5*self.density*(hor_velocity**2 + ver_velocity**2)
        return kinetic_energy_density*fdrk.dx

    def hamiltonian(self, hor_displacement, ver_displacement, hor_velocity, ver_velocity):
        return self.kinetic_energy(hor_velocity, ver_velocity) + self.deformation_energy(hor_displacement, ver_displacement)
    

    def deformation_energy_leapfrog(self, hor_displacement_min, hor_displacement_plus, ver_displacement_min, ver_displacement_plus):
        axial_energy_density = 0.5*self.axial_stiffness*fdrk.inner(hor_displacement_min.dx(0), hor_displacement_plus.dx(0))
        bending_energy_density = 0.5*self.bending_stiffness*fdrk.inner(ver_displacement_min.dx(0).dx(0), ver_displacement_plus.dx(0).dx(0))
        deformation_density = axial_energy_density + bending_energy_density

        return deformation_density*fdrk.dx 
        

    def weak_grad_potential(self, test_hor_velocity, test_ver_velocity, hor_displacement, ver_displacement):
        N_xx = self.axial_stress(hor_displacement, ver_displacement)
        bend_mom = self.bending_stress(ver_displacement)

        form_dV_hor_displacement = + fdrk.inner(test_hor_velocity.dx(0), N_xx)*fdrk.dx 
        
        form_dV_ver_displacement = + fdrk.inner(test_ver_velocity.dx(0).dx(0), bend_mom)*fdrk.dx \
                      + fdrk.inner(test_ver_velocity.dx(0), N_xx*ver_displacement.dx(0))*fdrk.dx
        
        return form_dV_hor_displacement, form_dV_ver_displacement
    

    def weak_grad_potential_linear(self, test_hor_velocity, test_ver_velocity, hor_displacement, ver_displacement):
        N_xx = self.axial_stiffness*hor_displacement.dx(0)
        bend_mom = self.bending_stress(ver_displacement)

        form_dV_hor_displacement = + fdrk.inner(test_hor_velocity.dx(0), N_xx)*fdrk.dx 
        
        form_dV_ver_displacement = + fdrk.inner(test_ver_velocity.dx(0).dx(0), bend_mom)*fdrk.dx 
        
        return form_dV_hor_displacement, form_dV_ver_displacement
    

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
        bc_hor_velocity = fdrk.DirichletBC(self.space_hor_velocity, fdrk.Constant(0), "on_boundary")

        bc_ver_displacement = fdrk.DirichletBC(self.space_ver_displacement, fdrk.Constant(0), "on_boundary")
        bc_ver_velocity = fdrk.DirichletBC(self.space_ver_velocity, fdrk.Constant(0), "on_boundary")

        hor_displacement_old = fdrk.Function(self.space_hor_displacement)
        hor_velocity_old = fdrk.Function(self.space_hor_velocity)

        ver_displacement_old = fdrk.Function(self.space_ver_displacement)
        ver_velocity_old = fdrk.Function(self.space_ver_velocity)

        ver_displacement_0 = self.get_initial_conditions_ver_displacement()
        ver_displacement_old.assign(fdrk.project(ver_displacement_0, self.space_ver_displacement, bcs = bc_ver_displacement))

        hor_displacement_new = fdrk.Function(self.space_hor_displacement)
        ver_displacement_new = fdrk.Function(self.space_ver_displacement)

        hor_velocity_new = fdrk.Function(self.space_hor_velocity)
        ver_velocity_new = fdrk.Function(self.space_ver_velocity)

        test_hor_velocity = fdrk.TestFunction(self.space_hor_velocity)
        test_ver_velocity = fdrk.TestFunction(self.space_ver_velocity)
        trial_hor_velocity = fdrk.TrialFunction(self.space_hor_velocity)
        trial_ver_velocity = fdrk.TrialFunction(self.space_ver_velocity)

        hor_displacement_half = fdrk.Function(self.space_hor_displacement)
        hor_displacement_half.assign(hor_displacement_old + 0.5*self.dt*hor_velocity_old)

        ver_displacement_half = fdrk.Function(self.space_ver_displacement)
        ver_displacement_half.assign(ver_displacement_old + 0.5*self.dt*ver_velocity_old)

        hor_displacement_new_half = hor_displacement_half + self.dt*hor_velocity_new
        ver_displacement_new_half = ver_displacement_half + self.dt*ver_velocity_new

        dV_hor_displacement, dV_ver_displacement = self.weak_grad_potential(test_hor_velocity, test_ver_velocity, hor_displacement_half, ver_displacement_half)
        # dV_hor_displacement, dV_ver_displacement = self.weak_grad_potential_linear(test_hor_velocity, test_ver_velocity, hor_displacement_half, ver_displacement_half)

        mass_hor_velocity = self.mass_form(test_hor_velocity, trial_hor_velocity)
        rhs_hor_velocity  = self.mass_form(test_hor_velocity, hor_velocity_old) - self.dt * dV_hor_displacement

        problem_hor_velocity = fdrk.LinearVariationalProblem(mass_hor_velocity, rhs_hor_velocity, hor_velocity_new, bcs=bc_hor_velocity)
        solver_hor_velocity = fdrk.LinearVariationalSolver(problem_hor_velocity)

        mass_ver_velocity = self.mass_form(test_ver_velocity, trial_ver_velocity)
        rhs_ver_velocity = self.mass_form(test_ver_velocity, ver_velocity_old) - self.dt * dV_ver_displacement

        problem_ver_velocity = fdrk.LinearVariationalProblem(mass_ver_velocity, rhs_ver_velocity, ver_velocity_new, bcs=bc_ver_velocity)
        
        solver_ver_velocity = fdrk.LinearVariationalSolver(problem_ver_velocity)

        hor_displacement_list = []
        ver_displacement_list = []
        hor_velocity_list = []
        ver_velocity_list = []

        if save_vars:   
            hor_displacement_list.append(hor_displacement_old.copy(deepcopy=True))
            ver_displacement_list.append(ver_displacement_old.copy(deepcopy=True))
            hor_velocity_list.append(hor_velocity_old.copy(deepcopy=True))
            ver_velocity_list.append(ver_velocity_old.copy(deepcopy=True))

        # energy_vec = np.zeros(len(self.t_vec_output)-1)
        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(hor_displacement_old, ver_displacement_old, hor_velocity_old, ver_velocity_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):
            solver_hor_velocity.solve()
            solver_ver_velocity.solve()
            
            hor_displacement_new.assign(0.5*(hor_displacement_half + hor_displacement_new_half))      
            ver_displacement_new.assign(0.5*(ver_displacement_half + ver_displacement_new_half))

             
            if (ii+1)%self.output_frequency==0:
                # energy_vec[kk] = fdrk.assemble(self.kinetic_energy(hor_velocity_new, ver_velocity_new) \
                #                 + self.deformation_energy_leapfrog(hor_displacement_half, hor_displacement_new_half, ver_displacement_half, ver_displacement_new_half))
                kk += 1
                energy_vec[kk] = fdrk.assemble(self.hamiltonian(hor_displacement_new, ver_displacement_new, hor_velocity_new, ver_velocity_new))
                if save_vars: 
                    hor_displacement_list.append(hor_displacement_new.copy(deepcopy=True))
                    ver_displacement_list.append(ver_displacement_new.copy(deepcopy=True))
                    hor_velocity_list.append(hor_velocity_new.copy(deepcopy=True))
                    ver_velocity_list.append(ver_velocity_new.copy(deepcopy=True))

            hor_displacement_half.assign(hor_displacement_new_half)
            hor_velocity_old.assign(hor_velocity_new)

            ver_displacement_half.assign(ver_displacement_new_half)
            ver_velocity_old.assign(ver_velocity_new)
           
        dict_results = {"hor_displacement": hor_displacement_list, 
                        "hor_velocity": hor_velocity_list, 
                        "ver_displacement": ver_displacement_list, 
                        "ver_velocity": ver_velocity_list, 
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
        bc_hor_displacement = fdrk.DirichletBC(self.mixed_space_implicit.sub(0), fdrk.Constant(0), "on_boundary")
        bc_ver_displacement = fdrk.DirichletBC(self.mixed_space_implicit.sub(1), fdrk.Constant(0), "on_boundary")
        bc_hor_velocity = fdrk.DirichletBC(self.mixed_space_implicit.sub(2), fdrk.Constant(0), "on_boundary")
        bc_ver_velocity = fdrk.DirichletBC(self.mixed_space_implicit.sub(3), fdrk.Constant(0), "on_boundary")

        bcs = [bc_hor_displacement, bc_ver_displacement, bc_hor_velocity, bc_ver_velocity]

        test_hor_displacement, test_ver_displacement, test_hor_velocity, test_ver_velocity = fdrk.TestFunctions(self.mixed_space_implicit)

        state_old = fdrk.Function(self.mixed_space_implicit) 
        hor_displacement_old, ver_displacement_old, hor_velocity_old, ver_velocity_old = state_old.subfunctions

        ver_displacement_0 = self.get_initial_conditions_ver_displacement()
        ver_displacement_old.assign(fdrk.project(ver_displacement_0, self.space_ver_displacement, \
                    fdrk.DirichletBC(self.space_ver_displacement, fdrk.Constant(0), "on_boundary")))

        state_new = fdrk.Function(self.mixed_space_implicit)
        hor_displacement_new, ver_displacement_new, hor_velocity_new, ver_velocity_new = fdrk.split(state_new)

        perturbation = fdrk.Constant(1e-3)
        state_new.sub(0).assign(perturbation)
        proj_perturbation = fdrk.project(perturbation, self.space_ver_displacement)
        state_new.sub(1).assign(ver_displacement_old + proj_perturbation)

        hor_velocity_midpoint = 0.5*(hor_velocity_old + hor_velocity_new)
        ver_velocity_midpoint = 0.5*(ver_velocity_old + ver_velocity_new)

        if type == "implicit midpoint":
            hor_displacement_midpoint = 0.5*(hor_displacement_old + hor_displacement_new)
            ver_displacement_midpoint = 0.5*(ver_displacement_old + ver_displacement_new)

            dV_hor_displacement, dV_ver_displacement = self.weak_grad_potential(test_hor_velocity, test_ver_velocity, \
                                                                                hor_displacement_midpoint, ver_displacement_midpoint)
        elif type == "discrete gradient":
            q_new = (hor_displacement_new, ver_displacement_new)
            q_old = (hor_displacement_old, ver_displacement_old)
            q_test = (test_hor_velocity, test_ver_velocity)
            dV_hor_displacement, dV_ver_displacement = discrete_gradient_firedrake_twofield(q_new, q_old, q_test, \
                                                self.weak_grad_potential, self.deformation_energy)

        else:
            raise ValueError("Unknown type of implicit method")
        
        res_hor_displacement = fdrk.inner(test_hor_displacement, hor_displacement_new - hor_displacement_old - self.dt * hor_velocity_midpoint)*fdrk.dx
        res_ver_displacement = fdrk.inner(test_ver_displacement, ver_displacement_new - ver_displacement_old - self.dt * ver_velocity_midpoint)*fdrk.dx
        res_hor_velocity = fdrk.inner(test_hor_velocity, hor_velocity_new - hor_velocity_old)*fdrk.dx + self.dt * dV_hor_displacement
        res_ver_velocity = fdrk.inner(test_ver_velocity, ver_velocity_new - ver_velocity_old)*fdrk.dx + self.dt * dV_ver_displacement

        residual = res_hor_displacement + res_ver_displacement + res_hor_velocity + res_ver_velocity

        
        problem = fdrk.NonlinearVariationalProblem(residual, state_new, bcs=bcs)
        solver = fdrk.NonlinearVariationalSolver(problem)

        hor_displacement_list = []
        ver_displacement_list = []
        hor_velocity_list = []
        ver_velocity_list = []

        if save_vars:   
            hor_displacement_list.append(hor_displacement_old.copy(deepcopy=True))
            ver_displacement_list.append(ver_displacement_old.copy(deepcopy=True))
            hor_velocity_list.append(hor_velocity_old.copy(deepcopy=True))
            ver_velocity_list.append(ver_velocity_old.copy(deepcopy=True))

        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(hor_displacement_old, ver_displacement_old, hor_velocity_old, ver_velocity_old))
        kk = 0

        for ii in tqdm(range(self.n_steps)):
            solver.solve()
            state_old.assign(state_new)

            state_new.sub(0).assign(hor_displacement_old + fdrk.interpolate(perturbation, self.space_hor_displacement))
            state_new.sub(1).assign(ver_displacement_old + proj_perturbation)

            # if fdrk.norm(state_new.sub(0))<perturbation:
            #     state_new.sub(0).assign(perturbation)
            # if fdrk.norm(state_new.sub(1))<perturbation:
            #     state_new.sub(1).assign(perturbation)

            if (ii+1)%self.output_frequency==0:
                kk += 1
                energy_vec[kk] = fdrk.assemble(self.hamiltonian(hor_displacement_old, ver_displacement_old, hor_velocity_old, ver_velocity_old))
                if save_vars: 
                    hor_displacement_list.append(hor_displacement_old.copy(deepcopy=True))
                    ver_displacement_list.append(ver_displacement_old.copy(deepcopy=True))
                    hor_velocity_list.append(hor_velocity_old.copy(deepcopy=True))
                    ver_velocity_list.append(ver_velocity_old.copy(deepcopy=True))


        dict_results = {"hor_displacement": hor_displacement_list, 
                        "hor_velocity": hor_velocity_list, 
                        "ver_displacement": ver_displacement_list, 
                        "ver_velocity": ver_velocity_list, 
                        "energy": energy_vec}
        
        return dict_results