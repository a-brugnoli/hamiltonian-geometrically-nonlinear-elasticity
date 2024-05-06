# Implementation of mixed finite elements for von Karman plate
import firedrake as fdrk
import os 
from src.postprocessing.animators import animate_scalar_trisurf, animate_scalar_tripcolor
from src.tools.common import compute_min_max_function
from tqdm import tqdm
from math import ceil

import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

# Matplotlib settings
import matplotlib
import matplotlib.pyplot as plt

save_res = True

# coefficients 
h = fdrk.Constant(0.002)
nu = fdrk.Constant(0.3)
rho = fdrk.Constant(7850)
E = fdrk.Constant(2 * 10**11)

D_bend = fdrk.Constant(E * h ** 3 / (1 - nu ** 2) / 12)
C_bend = fdrk.Constant(12 / (E * h ** 3))

D_men = fdrk.Constant(E * h / (1 - nu ** 2))
C_men = fdrk.Constant(1 / (E * h))


# Operators and functions
def gradSym(u):
    return 0.5 * (fdrk.nabla_grad(u) + fdrk.nabla_grad(u).T)


def membrane_stiff(eps_0):
    n_stress = D_men * ((1 - nu) * eps_0 + nu * fdrk.Identity(2) * fdrk.tr(eps_0))
    return n_stress

def membrane_comp(n_stress):
    eps_0 = C_men * ((1 + nu)*n_stress- nu * fdrk.Identity(2) * fdrk.tr(n_stress))
    return eps_0

def bending_stiff(curv):
    m_stress = D_bend * ((1 - nu) * curv + nu * fdrk.Identity(2) * fdrk.tr(curv))
    return m_stress


def bending_comp(m_stress):
    curv = C_bend * ((1 + nu)*m_stress - nu * fdrk.Identity(2) * fdrk.tr(m_stress))
    return curv


    
def m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                   e_u, e_eps, e_w, e_kap, e_disp):
        
        al_u = rho * h * e_u
        al_eps = membrane_comp(e_eps)
        
        al_w = rho * h * e_w
        al_kap = bending_comp(e_kap)
        
        m_form = fdrk.inner(v_u, al_u) * fdrk.dx \
               + fdrk.inner(v_eps, al_eps) * fdrk.dx \
               + fdrk.inner(v_w, al_w) * fdrk.dx \
               + fdrk.inner(v_kap, al_kap) * fdrk.dx \
               + fdrk.inner(v_disp, e_disp) * fdrk.dx
               
        return m_form
    
    
def j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
               e_u, e_eps, e_w, e_kap, e_disp, mesh):
    
    n_ver = fdrk.FacetNormal(mesh)
    s_ver = fdrk.as_vector([-n_ver[1], n_ver[0]])
    
    j_axial = fdrk.inner(v_eps, gradSym(e_u)) * fdrk.dx \
            - fdrk.inner(gradSym(v_u), e_eps) * fdrk.dx
            
    j_bend = - fdrk.inner(fdrk.grad(fdrk.grad(v_w)), e_kap) * fdrk.dx \
        + fdrk.inner(v_kap, fdrk.grad(fdrk.grad(e_w))) * fdrk.dx \
        + fdrk.jump(fdrk.grad(v_w), n_ver) * fdrk.dot(fdrk.dot(e_kap('+'), n_ver('+')), n_ver('+')) * fdrk.dS \
        + fdrk.dot(fdrk.grad(v_w), n_ver) * fdrk.dot(fdrk.dot(e_kap, n_ver), n_ver) * fdrk.ds \
        - fdrk.dot(fdrk.dot(v_kap('+'), n_ver('+')), n_ver('+')) * fdrk.jump(fdrk.grad(e_w), n_ver) * fdrk.dS \
        - fdrk.dot(fdrk.dot(v_kap, n_ver), n_ver) * fdrk.dot(fdrk.grad(e_w), n_ver) * fdrk.ds
    
    j_coup = fdrk.inner(v_eps, fdrk.sym(fdrk.outer(fdrk.grad(e_disp), fdrk.grad(e_w)))) * fdrk.dx \
           - fdrk.inner(fdrk.sym(fdrk.outer(fdrk.grad(v_w), fdrk.grad(e_disp))),  e_eps) * fdrk.dx
    
    m_w = fdrk.inner(v_disp, e_w) * fdrk.dx
    
    j_form = j_axial + j_bend + j_coup + m_w
    
    return j_form 



def compute_sol(n_elem, deg, amplitude):

    L_x, L_y = 0.5, 0.5
    mesh = fdrk.RectangleMesh(n_elem, n_elem, L_x, L_y, quadrilateral=False)
    
    V_u = fdrk.VectorFunctionSpace(mesh, "CG", deg)
    V_eps = fdrk.FunctionSpace(mesh, "Regge", deg-1)

    V_w = fdrk.FunctionSpace(mesh, "CG", deg)
    V_kap = fdrk.FunctionSpace(mesh, "HHJ", deg-1)
    V_disp = fdrk.FunctionSpace(mesh, "CG", deg)
    
    
    V = V_u * V_eps * V_w * V_kap * V_disp
    
    print("Number of dofs: " +  str(V.dim()))
    
    v_u, v_eps, v_w, v_kap, v_disp = fdrk.TestFunctions(V)
    

    dx = fdrk.Measure('dx')
    ds = fdrk.Measure('ds')
    dS = fdrk.Measure("dS")    
    
    
    bcs = []
    bc_u = fdrk.DirichletBC(V.sub(0), fdrk.Constant((0.0, 0.0)), "on_boundary")
    bc_w = fdrk.DirichletBC(V.sub(2), fdrk.Constant(0.0), "on_boundary")
    bc_kap = fdrk.DirichletBC(V.sub(3), fdrk.Constant(((0.0, 0.0), (0.0, 0.0))), "on_boundary")
    bc_disp = fdrk.DirichletBC(V.sub(4), fdrk.Constant(0.0), "on_boundary")
    
    bcs.append(bc_u)
    bcs.append(bc_w)
    bcs.append(bc_kap)
    bcs.append(bc_disp)

    t = 0.
    T_end = 7.5 * 10**(-3)        # total simulation time
    
    time_step = 5 * 10**(-6)
    t_ = fdrk.Constant(t)
    t_1 = fdrk.Constant(t+time_step)
    theta = 0.5
    
    x, y = mesh.coordinates
    
    w_0 = amplitude * h * fdrk.sin(pi * x/ L_x) * fdrk.sin(pi * y/ L_y)

    bend_strain_0 = gradSym(fdrk.grad(w_0))
    bend_stress_0 = bending_stiff(bend_strain_0)

    mem_strain_0 = 0.5 * fdrk.outer(fdrk.grad(w_0), fdrk.grad(w_0))
    mem_stress_0 = membrane_stiff(mem_strain_0)

    e_n = fdrk.Function(V,  name="e old")
    e_n1 = fdrk.Function(V,  name="e new")

    e_n.sub(1).assign(fdrk.interpolate(mem_stress_0, V_eps))
    e_n.sub(3).assign(fdrk.interpolate(bend_stress_0, V_kap))
    e_n.sub(4).assign(fdrk.interpolate(w_0, V_disp))
    
    e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n = e_n.subfunctions
    e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1 = fdrk.split(e_n1)

    left_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
              e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1) \
            - time_step*theta*j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1, mesh)

    right_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
            e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n) \
            + time_step * (1 - theta) * j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                                e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n, mesh) 

        
    n_time  = ceil(T_end/time_step)

    output_frequency = 10

    min_max_vel = (0, 0)
    min_max_disp = (0, 0)

    time_frames_ms = []
    time_frames_ms.append(0)

    list_frames_bend_velocity = []
    list_frames_bend_velocity.append(e_w_n.copy(deepcopy=True))

    list_frames_bend_displacement = []
    list_frames_bend_displacement.append(e_disp_n.copy(deepcopy=True))

    directory_largedata = "/home/dmsm/a.brugnoli/StoreResults/VonKarman/"
    if not os.path.exists(directory_largedata):
        os.makedirs(directory_largedata, exist_ok=True)
    
    outfile_bend_velocity = fdrk.File(f"{directory_largedata}\
                                    /NonLinear/amp_{amplitude}/Vertical_velocity.pvd")
    outfile_bend_velocity.write(e_w_n, time=0)

    outfile_bend_displacement = fdrk.File(f"{directory_largedata}\
                                    /NonLinear/amp_{amplitude}/Vertical_displacement.pvd")
    outfile_bend_displacement.write(e_disp_n, time=0)

    param = {'snes_type': 'newtonls', 'ksp_type': 'preonly', 'pc_type': 'lu'} 
             # 'snes_rtol': '1e+1', 'snes_atol': '1e-10','snes_stol': '1e+1', 
             # 'snes_max_it': '50', 'ksp_rtol': '1e+1', 'ksp_atol': '1e-10', 
             # 'ksp_rtol': '1e+1', 'ksp_atol': '1e-10', 'ksp_divtol': '1e15'}

    residual = left_hs - right_hs
    nonlinear_problem = fdrk.NonlinearVariationalProblem(residual, 
                                                        e_n1, 
                                                        bcs=bcs)

    solver = fdrk.NonlinearVariationalSolver(nonlinear_problem,
                                            solver_parameters=param)
    
        
    for ii in tqdm(range(1, n_time+1)):
        actual_time = ii*time_step

        e_n1.assign(e_n) #  For initialisation
        solver.solve()
        
        e_n.assign(e_n1)
        
        t += time_step
        t_.assign(t)
        t_1.assign(t+time_step)
  

        if ii % output_frequency == 0:

            min_max_vel = compute_min_max_function(e_w_n, min_max_vel)
            min_max_disp = compute_min_max_function(e_disp_n, min_max_disp)

            list_frames_bend_velocity.append(e_w_n.copy(deepcopy=True))
            list_frames_bend_displacement.append(e_disp_n.copy(deepcopy=True))
            time_frames_ms.append(10**3 * actual_time)

            outfile_bend_velocity.write(e_w_n, time=actual_time)
            outfile_bend_displacement.write(e_disp_n, time=actual_time)


    directory_results = f"{os.path.dirname(os.path.abspath(__file__))}/results/VonKarmanNonLinear/FirstMode/amp_{amplitude}/"
    if not os.path.exists(directory_results):
        os.makedirs(directory_results, exist_ok=True)
    interval = 10e5 * output_frequency * time_step

    velocity_animation = animate_scalar_trisurf(time_frames_ms, list_frames_bend_velocity,\
                                                interval=interval, lim_z = min_max_vel)

    velocity_animation.save(f"{directory_results}Animation_velocity.mp4", writer="ffmpeg")

    displacement_animation = animate_scalar_trisurf(time_frames_ms, list_frames_bend_displacement,\
                                                interval=interval, lim_z = min_max_disp)

    displacement_animation.save(f"{directory_results}Animation_displacement.mp4", writer="ffmpeg")

    n_frames = len(time_frames_ms)

    indexes_images = [0, int(n_frames/3), int(2*n_frames/3), int(n_frames-1)]

    print(n_frames, indexes_images)

    for kk in indexes_images:
        time_image = time_frames_ms[kk]

        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.set_aspect('equal')
        fdrk.trisurf(list_frames_bend_displacement[kk], axes=axes)
        axes.set_title(f"Displacement $t={time_image:.1f}$ [ms]", loc='center')
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.set_zlim(min_max_disp)

        plt.savefig(f"{directory_results}Displacement_t{time_image:.1f}.pdf", bbox_inches='tight', dpi='figure', format='pdf')


    return 


n_elem = 10
deg = 1
amplitude = 2
compute_sol(n_elem, deg, amplitude)


