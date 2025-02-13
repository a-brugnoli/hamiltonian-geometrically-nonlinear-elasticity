import firedrake as fdrk
import numpy as np
from tqdm import tqdm
from src.discrete_gradient import discrete_gradient_firedrake

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
        self.space_q_x = fdrk.FunctionSpace(self.domain, "CG", 1)
        self.space_v_x = self.space_q_x

        self.space_q_z = fdrk.FunctionSpace(self.domain, "Hermite", 3)
        self.space_v_z = self.space_q_z

        self.space_axial_stress = fdrk.FunctionSpace(self.domain, "DG", 0)
        self.space_bending_stress = fdrk.FunctionSpace(self.domain, "DG", 1)

        self.mixed_space_implicit = self.space_q_x * self.space_q_z * self.space_v_x * self.space_v_z 
        self.mixed_space_linear_implicit = self.space_v_x * self.space_v_z * self.space_axial_stress * self.space_bending_stress


    def get_initial_conditions_q_z(self):
        q_z_exp = self.q0*fdrk.sin(fdrk.pi*self.x_coord/self.length)
        return q_z_exp


    def axial_strain(self, q_x, q_z):
        try:
            eps_a = q_x.dx(0) + 0.5**(q_z.dx(0))**2
            return eps_a
        except ValueError:
            print("Invalid expression in axial strain. Don't use numbers.")
            return

    def bending_strain(self, q_z):
        try:
            kappa = q_z.dx(0).dx(0) 
        except ValueError:
            print("Invalid expression in bending strain. Don't use numbers.")
        return kappa
    
    
    def axial_stress(self, q_x, q_z):
        return self.axial_stiffness*self.axial_strain(q_x, q_z)
    
    def bending_stress(self, q_z):
        return self.bending_stiffness*self.bending_strain(q_z)
    

    def deformation_energy(self, q_x, q_z):
        potential_energy_density = 0.5*(self.axial_stiffness*self.axial_strain(q_x, q_z)**2 \
                                      + self.bending_stiffness*self.bending_strain(q_z)**2)
        V = potential_energy_density*fdrk.dx

        return V

    def kinetic_energy(self, v_x, v_z):
        kinetic_energy_density = 0.5*self.density*(v_x**2 + v_z**2)
        T = kinetic_energy_density*fdrk.dx

        return T

    def hamiltonian(self, q_x, q_z, v_x, v_z):
        return self.kinetic_energy(v_x, v_z) + self.deformation_energy(q_x, q_z)
    

    def deformation_energy_leapfrog(self, q_x_min, q_x_plus, q_z_min, q_z_plus):
        axial_energy_density = 0.5*self.axial_stiffness*fdrk.inner(q_x_min.dx(0), q_x_plus.dx(0))
        bending_energy_density = 0.5*self.bending_stiffness*fdrk.inner(q_z_min.dx(0).dx(0), q_z_plus.dx(0).dx(0))
        deformation_density = axial_energy_density + bending_energy_density

        return deformation_density*fdrk.dx 
        

    def weak_grad_potential(self, test_v_x, test_v_z, q_x, q_z):
        n_xx = self.axial_stress(q_x, q_z)
        m_xx = self.bending_stress(q_z)

        form_dV_q_x = + fdrk.inner(test_v_x.dx(0), n_xx)*fdrk.dx 
        
        form_dV_q_z = + fdrk.inner(test_v_z.dx(0).dx(0), m_xx)*fdrk.dx \
                      + fdrk.inner(test_v_z.dx(0), q_x.dx(0)*n_xx)*fdrk.dx
        
        return form_dV_q_x, form_dV_q_z
    

    def weak_grad_potential_linear(self, test_v_x, test_v_z, q_x, q_z):
        n_xx = self.axial_stiffness*q_x.dx(0)
        m_xx = self.bending_stress(q_z)

        form_dV_q_x = + fdrk.inner(test_v_x.dx(0), n_xx)*fdrk.dx 
        
        form_dV_q_z = + fdrk.inner(test_v_z.dx(0).dx(0), m_xx)*fdrk.dx 
        
        return form_dV_q_x, form_dV_q_z
    

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
        bc_v_x = fdrk.DirichletBC(self.space_v_x, fdrk.Constant(0), "on_boundary")

        bc_q_z = fdrk.DirichletBC(self.space_q_z, fdrk.Constant(0), "on_boundary")
        bc_v_z = fdrk.DirichletBC(self.space_v_z, fdrk.Constant(0), "on_boundary")

        q_x_old = fdrk.Function(self.space_q_x)
        v_x_old = fdrk.Function(self.space_v_x)

        q_z_old = fdrk.Function(self.space_q_z)
        v_z_old = fdrk.Function(self.space_v_z)

        q_z_0 = self.get_initial_conditions_q_z()
        q_z_old.assign(fdrk.project(q_z_0, self.space_q_z, bcs = bc_q_z))

        q_x_new = fdrk.Function(self.space_q_x)
        q_z_new = fdrk.Function(self.space_q_z)

        v_x_new = fdrk.Function(self.space_v_x)
        v_z_new = fdrk.Function(self.space_v_z)

        test_v_x = fdrk.TestFunction(self.space_v_x)
        test_v_z = fdrk.TestFunction(self.space_v_z)
        trial_v_x = fdrk.TrialFunction(self.space_v_x)
        trial_v_z = fdrk.TrialFunction(self.space_v_z)

        q_x_half = fdrk.Function(self.space_q_x)
        q_x_half.assign(q_x_old + 0.5*self.dt*v_x_old)

        q_z_half = fdrk.Function(self.space_q_z)
        q_z_half.assign(q_z_old + 0.5*self.dt*v_z_old)

        q_x_new_half = q_x_half + self.dt*v_x_new
        q_z_new_half = q_z_half + self.dt*v_z_new

        dV_q_x, dV_q_z = self.weak_grad_potential(test_v_x, test_v_z, q_x_half, q_z_half)
        # dV_q_x, dV_q_z = self.weak_grad_potential_linear(test_v_x, test_v_z, q_x_half, q_z_half)

        mass_v_x = self.mass_form(test_v_x, trial_v_x)
        rhs_v_x  = self.mass_form(test_v_x, v_x_old) - self.dt * dV_q_x

        problem_v_x = fdrk.LinearVariationalProblem(mass_v_x, rhs_v_x, v_x_new, bcs=bc_v_x)
        solver_v_x = fdrk.LinearVariationalSolver(problem_v_x)

        mass_v_z = self.mass_form(test_v_z, trial_v_z)
        rhs_v_z = self.mass_form(test_v_z, v_z_old) - self.dt * dV_q_z

        problem_v_z = fdrk.LinearVariationalProblem(mass_v_z, rhs_v_z, v_z_new, bcs=bc_v_z)
        
        solver_v_z = fdrk.LinearVariationalSolver(problem_v_z)

        q_x_list = []
        q_z_list = []
        v_x_list = []
        v_z_list = []

        if save_vars:   
            q_x_list.append(q_x_old.copy(deepcopy=True))
            q_z_list.append(q_z_old.copy(deepcopy=True))
            v_x_list.append(v_x_old.copy(deepcopy=True))
            v_z_list.append(v_z_old.copy(deepcopy=True))

        energy_vec = np.zeros(len(self.t_vec_output))
        energy_vec[0] = fdrk.assemble(self.hamiltonian(q_x_old, q_z_old, v_x_old, v_z_old))
        kk = 0
        # energy_vec_leapfrog = np.zeros(self.n_steps)

        for ii in tqdm(range(self.n_steps)):
            solver_v_x.solve()
            solver_v_z.solve()
            
            q_x_new.assign(0.5*(q_x_half + q_x_new_half))      
            q_z_new.assign(0.5*(q_z_half + q_z_new_half))


            # energy_vec_leapfrog[ii] = fdrk.assemble(self.kinetic_energy(v_x_new, v_z_new) \
            #                     + self.deformation_energy_leapfrog(q_x_half, q_x_new_half, q_z_half, q_z_new_half))

            if (ii+1)%self.output_frequency==0:
                kk += 1
                energy_vec[kk] = fdrk.assemble(self.hamiltonian(q_x_new, q_z_new, v_x_new, v_z_new))
                if save_vars: 
                    q_x_list.append(q_x_new.copy(deepcopy=True))
                    q_z_list.append(q_z_new.copy(deepcopy=True))
                    v_x_list.append(v_x_new.copy(deepcopy=True))
                    v_z_list.append(v_z_new.copy(deepcopy=True))

            q_x_half.assign(q_x_new_half)
            v_x_old.assign(v_x_new)

            q_z_half.assign(q_z_new_half)
            v_z_old.assign(v_z_new)

        dict_results = {"q_x": q_x_list, 
                        "v_x": v_x_list, 
                        "q_z": q_z_list, 
                        "v_z": v_z_list, 
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
        bc_q_x = fdrk.DirichletBC(self.space_v_x, fdrk.Constant(0), "on_boundary")
        bc_v_x = fdrk.DirichletBC(self.space_v_x, fdrk.Constant(0), "on_boundary")

        bc_q_z = fdrk.DirichletBC(self.space_q_z, fdrk.Constant(0), "on_boundary")
        bc_v_z = fdrk.DirichletBC(self.space_v_z, fdrk.Constant(0), "on_boundary")

        bcs = [bc_q_x, bc_q_z, bc_v_x, bc_v_z]

        test_function = fdrk.TestFunctions(self.mixed_space_implicit)
        test_q_x, test_q_z, test_v_x, test_v_z = fdrk.TestFunctions(self.mixed_space_implicit)

        x_old = fdrk.Function(self.mixed_space_implicit) 
        q_x_old, q_z_old, v_x_old, v_z_old = x_old.subfunctions

        q_z_0 = self.get_initial_conditions_q_z()
        q_z_old.assign(fdrk.project(q_z_0, self.mixed_space_implicit.sub(1), bcs = bc_q_z))

        x_new = fdrk.Function(self.mixed_space_implicit)
        q_x_new, q_z_new, v_x_new, v_z_new = fdrk.split(x_new)

        v_x_midpoint = 0.5*(v_x_old + v_x_new)
        v_z_midpoint = 0.5*(v_z_old + v_z_new)

        def residual():
            if type == "implicit midpoint":
                q_x_midpoint = 0.5*(q_x_old + q_x_new)
                q_z_midpoint = 0.5*(q_z_old + q_z_new)

                dV_q_x, dV_q_z = self.weak_grad_potential(test_v_x, test_v_z, \
                                                          q_x_midpoint, q_z_midpoint)
            elif type == "discrete gradient":
                raise NotImplementedError("Discrete gradient not implemented")      
            else:
                raise ValueError("Unknown type of implicit method")
            
            res_q_x = fdrk.inner(test_q_x, q_x_new - q_x_old - self.dt * v_x_midpoint)*fdrk.dx
            res_q_z = fdrk.inner(test_q_z, q_z_new - q_z_old - self.dt * v_z_midpoint)*fdrk.dx
            res_v_x = fdrk.inner(test_v_x, v_x_new - v_x_old)*fdrk.dx + self.dt * dV_q_x
            res_v_z = fdrk.inner(test_v_z,v_z_new - v_z_old)*fdrk.dx + self.dt * dV_q_z

            res = res_q_x + res_q_z + res_v_x + res_v_z
            return res

        # problem_v_x = fdrk.LinearVariationalProblem(mass_v_x, rhs_v_x, \
        #                                             v_x_new, bcs=bc_v_x)
        # solver_v_x = fdrk.LinearVariationalSolver(problem_v_x)

        # mass_v_z = self.mass_form(test_v_z, trial_v_z)
        # rhs_v_z = self.mass_form(test_v_z, v_z_old) - self.dt * dV_q_z

        # problem_v_z = fdrk.LinearVariationalProblem(mass_v_z, rhs_v_z,\
        #                                             v_z_new, bcs = bc_v_z)
        
        # solver_v_z = fdrk.LinearVariationalSolver(problem_v_z)

        # q_x_list = []
        # q_z_list = []
        # v_x_list = []
        # v_z_list = []

        # if save_vars:   
        #     q_x_list.append(q_x_old.copy(deepcopy=True))
        #     q_z_list.append(q_z_old.copy(deepcopy=True))
        #     v_x_list.append(v_x_old.copy(deepcopy=True))
        #     v_z_list.append(v_z_old.copy(deepcopy=True))

        # energy_vec = np.zeros(len(self.t_vec_output))
        # energy_vec[0] = fdrk.assemble(self.hamiltonian(q_x_old, q_z_old, v_x_old, v_z_old))
        # kk = 0
        # # energy_vec_leapfrog = np.zeros(self.n_steps)

        # for ii in tqdm(range(self.n_steps)):
        #     solver_v_x.solve()
        #     solver_v_z.solve()
            
        #     q_x_new.assign(0.5*(q_x_half + q_x_new_half))      
        #     q_z_new.assign(0.5*(q_z_half + q_z_new_half))


        #     # energy_vec_leapfrog[ii] = fdrk.assemble(self.kinetic_energy(v_x_new, v_z_new) \
        #     #                     + self.deformation_energy_leapfrog(q_x_half, q_x_new_half, q_z_half, q_z_new_half))

        #     if (ii+1)%self.output_frequency==0:
        #         kk += 1
        #         energy_vec[kk] = fdrk.assemble(self.hamiltonian(q_x_new, q_z_new, v_x_new, v_z_new))
        #         if save_vars: 
        #             q_x_list.append(q_x_new.copy(deepcopy=True))
        #             q_z_list.append(q_z_new.copy(deepcopy=True))
        #             v_x_list.append(v_x_new.copy(deepcopy=True))
        #             v_z_list.append(v_z_new.copy(deepcopy=True))

        #     q_x_half.assign(q_x_new_half)
        #     v_x_old.assign(v_x_new)

        #     q_z_half.assign(q_z_new_half)
        #     v_z_old.assign(v_z_new)

        # dict_results = {"q_x": q_x_list, 
        #                 "v_x": v_x_list, 
        #                 "q_z": q_z_list, 
        #                 "v_z": v_z_list, 
        #                 "energy": energy_vec}
        
        # return dict_results