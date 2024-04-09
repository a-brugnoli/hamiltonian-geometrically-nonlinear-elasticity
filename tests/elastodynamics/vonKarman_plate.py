# Implementation of mixed finite elements for von Karman plate
import firedrake as fdrk

import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

# Matplotlib settings
import matplotlib
import matplotlib.pyplot as plt
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['text.usetex'] = True


save_res = True
# Geometrical coefficients 

nu = fdrk.Constant(0.3)
h = fdrk.Constant(0.1)

L = 1

# Physical coefficients

rho = fdrk.Constant(2700)
E = fdrk.Constant(70 * 10**3)

D_bend = fdrk.Constant(E * h ** 3 / (1 - nu ** 2) / 12)
C_bend = fdrk.Constant(12 / (E * h ** 3))

D_men = fdrk.Constant(E * h / (1 - nu ** 2))
C_men = fdrk.Constant(1 / (E * h))


# Operators and functions
def gradSym(u):
    return 0.5 * (fdrk.nabla_grad(u) + fdrk.nabla_grad(u).T)


def traction_stiff(eps_0):
    n_stress = D_men * ((1 - nu) * eps_0 + nu * fdrk.Identity(2) * fdrk.tr(eps_0))
    return n_stress

def traction_comp(n_stress):
    eps_0 = C_men * ((1+nu)*n_stress- nu * fdrk.Identity(2) * fdrk.tr(n_stress))
    return eps_0

def bending_stiff(curv):
    m_stress = D_bend * ((1 - nu) * curv + nu * fdrk.Identity(2) * fdrk.tr(curv))
    return m_stress


def bending_comp(m_stress):
    curv = C_bend * ((1+nu)*m_stress - nu * fdrk.Identity(2) * fdrk.tr(m_stress))
    return curv


    
def m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                   e_u, e_eps, e_w, e_kap, e_disp):
        
        al_u = rho * h * e_u
        al_eps = traction_comp(e_eps)
        
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
        + fdrk.jump(fdrk.grad(v_w), n_ver) * fdrk.dot(fdrk.dot(e_kap('+'), n_ver('+')), n_ver('+')) * fdrk.dS \
        + fdrk.dot(fdrk.grad(v_w), n_ver) * fdrk.dot(fdrk.dot(e_kap, n_ver), n_ver) * fdrk.ds \
        + fdrk.inner(v_kap, fdrk.grad(fdrk.grad(e_w))) * fdrk.dx \
        - fdrk.dot(fdrk.dot(v_kap('+'), n_ver('+')), n_ver('+')) * fdrk.jump(fdrk.grad(e_w), n_ver) * fdrk.dS \
        - fdrk.dot(fdrk.dot(v_kap, n_ver), n_ver) * fdrk.dot(fdrk.grad(e_w), n_ver) * fdrk.ds
    
    j_coup = fdrk.inner(v_eps, fdrk.sym(fdrk.outer(fdrk.grad(e_disp), fdrk.grad(e_w)))) * fdrk.dx \
           - fdrk.inner(fdrk.sym(fdrk.outer(fdrk.grad(v_w), fdrk.grad(e_disp))),  e_eps) * fdrk.dx
    
    m_w = fdrk.inner(v_disp, e_w) * fdrk.dx
    
    j_form = j_axial + j_bend + j_coup + m_w
    
    return j_form 



def compute_err(n_elem, deg):

    mesh = fdrk.RectangleMesh(n_elem, n_elem, L, L, quadrilateral=False)
    
    deg_eps = 2*(deg-1)

    V_u = fdrk.VectorFunctionSpace(mesh, "CG", deg_eps+1)
    V_epsD = fdrk.VectorFunctionSpace(mesh, "DG", deg_eps)
    V_eps12 = fdrk.FunctionSpace(mesh, "DG", deg_eps)

    # V_u = VectorFunctionSpace(mesh, "CG", deg)
    # V_epsD = VectorFunctionSpace(mesh, "DG", deg-1)
    # V_eps12 = FunctionSpace(mesh, "DG", deg-1)
    
    V_w = fdrk.FunctionSpace(mesh, "CG", deg)
    V_kap = fdrk.FunctionSpace(mesh, "HHJ", deg-1)
    V_disp = fdrk.FunctionSpace(mesh, "CG", deg)
    
    
    V = V_u * V_epsD * V_eps12 * V_w * V_kap * V_disp
    
    print("Number of dofs: " +  str(V.dim()))
    
    v_u, v_epsD, v_eps12, v_w, v_kap, v_disp = fdrk.TestFunctions(V)
    
    
    v_eps = fdrk.as_tensor([[v_epsD[0], v_eps12],
                       [v_eps12, v_epsD[1]]
                       ])
    
    dx = fdrk.Measure('dx')
    ds = fdrk.Measure('ds')
    dS = fdrk.Measure("dS")    
    
    
    bcs = []
    bc_u = fdrk.DirichletBC(V.sub(0), fdrk.Constant((0.0, 0.0)), "on_boundary")
    bc_w = fdrk.DirichletBC(V.sub(3), fdrk.Constant(0.0), "on_boundary")
    bc_kap = fdrk.DirichletBC(V.sub(4), fdrk.Constant(((0.0, 0.0), (0.0, 0.0))), "on_boundary")
    
    bcs.append(bc_u)
    bcs.append(bc_w)
    bcs.append(bc_kap)
    
    t = 0.
    t_fin = 1        # total simulation time
    
    dt = 1/(2*pi*n_elem)
    t_ = fdrk.Constant(t)
    t_1 = fdrk.Constant(t+dt)
    theta = 0.5
    
    x, y = mesh.coordinates
    
    T_u = t_fin
    omega_u = 2*pi/T_u*t_fin
    T_w = t_fin
    omega_w = 2*pi/T_w*t_fin
    
    # omega_u = 1
    # omega_w = 1
    
    u_st = fdrk.as_vector([x**4*(1-(x/L)**4)*fdrk.sin(pi*y/L)**2,
                      fdrk.sin(pi*x/L)**2*y**4*(1-(y/L)**4)])
    
    grad_u_st = fdrk.as_tensor([[4*x**3*(1-2*(x/L)**4)*fdrk.sin(pi*y/L)**2, pi/L*x**4*(1-(x/L)**4)*fdrk.sin(2*pi*y/L)],
                           [pi/L*fdrk.sin(2*pi*x/L)*y**4*(1-(y/L)**4), fdrk.sin(pi*x/L)**2*4*y**3*(1-2*(y/L)**4)]
                           ])
    
    u_ex = u_st*fdrk.sin(omega_u*t_)    
    e_u_ex = omega_u*u_st*fdrk.cos(omega_u*t_)
    dtt_u_ex = - omega_u**2 * u_st*fdrk.sin(omega_u*t_)
    
    w_st = fdrk.sin(pi*x/L)*fdrk.sin(pi*y/L)
    
    grad_w_st = fdrk.as_vector([pi/L*fdrk.cos(pi*x/L)*fdrk.sin(pi*y/L),
                           pi/L*fdrk.sin(pi*x/L)*fdrk.cos(pi*y/L)])
    
    Hess_w_st = fdrk.as_tensor([[-(pi/L)**2*fdrk.sin(pi*x/L)*fdrk.sin(pi*y/L), (pi/L)**2*fdrk.cos(pi*x/L)*fdrk.cos(pi*y/L)],
                           [(pi/L)**2*fdrk.cos(pi*x/L)*fdrk.cos(pi*y/L), -(pi/L)**2*fdrk.sin(pi*x/L)*fdrk.sin(pi*y/L)]
                           ])
    
    w_ex = w_st * fdrk.sin(omega_w*t_)
    e_w_ex = omega_w * w_st * fdrk.cos(omega_w*t_)
    dtt_w_ex = - omega_w**2*w_st*fdrk.sin(omega_w*t_)
    
    e_eps_ex = traction_stiff(fdrk.sym(grad_u_st)* fdrk.sin(omega_u*t_) \
                              + 0.5 * fdrk.sin(omega_w*t_)**2* fdrk.outer(grad_w_st, grad_w_st))
        
    e_kap_ex = bending_stiff(Hess_w_st * fdrk.sin(omega_w*t_))
    
    f_u = rho*h*dtt_u_ex - fdrk.div(e_eps_ex)
    f_w = rho*h*dtt_w_ex + fdrk.div(fdrk.div(e_kap_ex)) - fdrk.div(fdrk.dot(e_eps_ex, grad_w_st*fdrk.sin(omega_w*t_)))
        
    # e_eps_ex = traction_stiff(gradSym(u_ex) + 0.5 * outer(grad(w_ex), grad(w_ex)))
    # e_kap_ex = bending_stiff(grad(grad(w_ex)))
    
    # f_u = rho*h*dtt_u_ex - div(e_eps_ex)
    # f_w = rho*h*dtt_w_ex + div(div(e_kap_ex)) - div(dot(e_eps_ex, grad(w_ex)))
    
    f_form = fdrk.inner(v_u, f_u)*dx + fdrk.inner(v_w, f_w)*dx
    
    u_ex1 = u_st*fdrk.sin(omega_u*t_1)    
    e_u_ex1 = omega_u*u_st*fdrk.cos(omega_u*t_1)
    dtt_u_ex1 = - omega_u**2*u_st*fdrk.sin(omega_u*t_1)
    
    w_ex1 = w_st*fdrk.sin(omega_w*t_1)
    e_w_ex1 = omega_w* w_st * fdrk.cos(omega_w*t_1)
    dtt_w_ex1 = - omega_w**2* w_st *fdrk.sin(omega_w*t_1)
    
    e_eps_ex1 = traction_stiff(fdrk.sym(grad_u_st)*fdrk.sin(omega_u*t_1) \
                              + 0.5 * fdrk.sin(omega_w*t_1)**2*fdrk.outer(grad_w_st, grad_w_st))
        
    e_kap_ex1 = bending_stiff(Hess_w_st * fdrk.sin(omega_w*t_1))
    
    f_u1 = rho*h*dtt_u_ex1 - fdrk.div(e_eps_ex1)
    f_w1 = rho*h*dtt_w_ex1 + fdrk.div(fdrk.div(e_kap_ex1)) - fdrk.div(fdrk.dot(e_eps_ex1, grad_w_st*fdrk.sin(omega_w*t_1)))
    
    # e_eps_ex1 = traction_stiff(gradSym(u_ex1) + 0.5 * outer(grad(w_ex1), grad(w_ex1)))
    # e_kap_ex1 = bending_stiff(grad(grad(w_ex1)))
    
    # f_u1 = rho*h*dtt_u_ex1 - div(e_eps_ex1)
    # f_w1 = rho*h*dtt_w_ex1 + div(div(e_kap_ex1)) - div(dot(e_eps_ex1, grad(w_ex1)))
       
    f_form1 = fdrk.inner(v_u, f_u1)*dx + fdrk.inner(v_w, f_w1)*dx
    
    
    e_n = fdrk.Function(V,  name="e old")
    e_n1 = fdrk.Function(V,  name="e new")
    
    e_n.sub(0).assign(fdrk.project(e_u_ex, V_u))
    e_n.sub(3).assign(fdrk.project(e_w_ex, V_w))
    
    e_u_n, e_epsD_n, e_eps12_n, e_w_n, e_kap_n, e_disp_n = e_n.split()
    
    e_eps_n = fdrk.as_tensor([[e_epsD_n[0], e_eps12_n],
                         [e_eps12_n, e_epsD_n[1]]
                         ])
    
    n_t = int(floor(t_fin/dt) + 1)
    t_vec = np.arange(start=0, stop=n_t*dt, step=dt)
    
    e_u_err_H1 = np.zeros((n_t,))
    e_eps_err_L2 = np.zeros((n_t,))
    
    e_w_err_H1 = np.zeros((n_t,))
    e_kap_err_L2 = np.zeros((n_t,))
    
    w_err_H1 = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    e_u_atP = np.zeros((n_t,))
    
    
    Ppoint = 3*L/4

    
    e_u_err_H1[0] = np.sqrt(fdrk.assemble(fdrk.inner(e_u_n - e_u_ex, e_u_n - e_u_ex) * dx
                  + fdrk.inner(gradSym(e_u_n) - gradSym(e_u_ex), gradSym(e_u_n) - gradSym(e_u_ex)) * dx))
    
    e_eps_err_L2[0] = np.sqrt(fdrk.assemble(fdrk.inner(e_eps_n - e_eps_ex, e_eps_n - e_eps_ex) * dx))
    
    e_w_err_H1[0] = np.sqrt(fdrk.assemble(fdrk.inner(e_w_n - e_w_ex, e_w_n - e_w_ex) * dx
                  + fdrk.inner(fdrk.grad(e_w_n) - fdrk.grad(e_w_ex), fdrk.grad(e_w_n) - fdrk.grad(e_w_ex)) * dx))
    
    e_kap_err_L2[0] = np.sqrt(fdrk.assemble(fdrk.inner(e_kap_n - e_kap_ex, e_kap_n - e_kap_ex) * dx))
    
    w_err_H1[0] = np.sqrt(fdrk.assemble(fdrk.inner(e_disp_n-w_ex, e_disp_n-w_ex) * dx
                + fdrk.inner(fdrk.grad(e_disp_n) - fdrk.grad(w_ex), fdrk.grad(e_disp_n) - fdrk.grad(w_ex)) * dx))
    
    param = {'snes_type': 'newtonls', 'ksp_type': 'preonly', 'pc_type': 'lu'} 
             # 'snes_rtol': '1e+1', 'snes_atol': '1e-10','snes_stol': '1e+1', 
             # 'snes_max_it': '50', 'ksp_rtol': '1e+1', 'ksp_atol': '1e-10', 
             # 'ksp_rtol': '1e+1', 'ksp_atol': '1e-10', 'ksp_divtol': '1e15'}
        
    for i in range(1, n_t):


        e_u_n, e_epsD_n, e_eps12_n, e_w_n, e_kap_n, e_disp_n = e_n.split()

        e_eps_n = fdrk.as_tensor([[e_epsD_n[0], e_eps12_n],
                             [e_eps12_n, e_epsD_n[1]]
                             ])
                
        e_u_n1, e_epsD_n1, e_eps12_n1, e_w_n1, e_kap_n1, e_disp_n1 = fdrk.split(e_n1)
       
        e_eps_n1 = fdrk.as_tensor([[e_epsD_n1[0], e_eps12_n1],
                              [e_eps12_n1, e_epsD_n1[1]]
                              ])
       
        left_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
              e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1) \
            - dt*theta*j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                       e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1, mesh)


        right_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
              e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n) \
              + dt * (1 - theta) * j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                                   e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n, mesh) \
              + dt * ((1 - theta) * f_form + theta * f_form1)

        
        F = left_hs - right_hs

        e_n1.assign(e_n) #  For initialisation
        fdrk.solve(F==0, e_n1, bcs=bcs, \
              solver_parameters=param)
        
        e_n.assign(e_n1)
        
        t += dt
        t_.assign(t)
        t_1.assign(t+dt)
  
        e_u_n, e_epsD_n, e_eps12_n, e_w_n, e_kap_n, e_disp_n = e_n.split()

        e_eps_n = fdrk.as_tensor([[e_epsD_n[0], e_eps12_n],
                             [e_eps12_n, e_epsD_n[1]]
                             ])
        

        e_u_err_H1[i] = np.sqrt(fdrk.assemble(fdrk.inner(e_u_n - e_u_ex, e_u_n - e_u_ex) * dx
                  + fdrk.inner(gradSym(e_u_n) - gradSym(e_u_ex), gradSym(e_u_n) - gradSym(e_u_ex)) * dx))
    
        e_eps_err_L2[i] = np.sqrt(fdrk.assemble(fdrk.inner(e_eps_n - e_eps_ex, e_eps_n - e_eps_ex) * dx))
        
        e_w_err_H1[i] = np.sqrt(fdrk.assemble(fdrk.inner(e_w_n - e_w_ex, e_w_n - e_w_ex) * dx
                      + fdrk.inner(fdrk.grad(e_w_n) - fdrk.grad(e_w_ex), fdrk.grad(e_w_n) - fdrk.grad(e_w_ex)) * dx))
        
        e_kap_err_L2[i] = np.sqrt(fdrk.assemble(fdrk.inner(e_kap_n - e_kap_ex, e_kap_n - e_kap_ex) * dx))
        
        w_err_H1[i] = np.sqrt(fdrk.assemble(fdrk.inner(e_disp_n-w_ex, e_disp_n-w_ex) * dx
                + fdrk.inner(fdrk.grad(e_disp_n) - fdrk.grad(w_ex), fdrk.grad(e_disp_n) - fdrk.grad(w_ex)) * dx))
    

        # w_atP[i] = e_disp_n.at(Ppoint)
    

    # plt.figure()
    # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # plt.plot(t_vec, np.sin(pi*Ppoint/L)*np.sin(omega_w*t_vec), 'b-', label=r'exact $w$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'Displacement at: ' + str(Ppoint))
    # plt.legend()
     
    # x_num, e_eps_num = calculate_one_dim_points(e_eps_n, 1) 

    # V_plot = FunctionSpace(mesh, "CG", 5)
    # x_an, e_eps_an = calculate_one_dim_points(interpolate(e_eps_ex, V_plot), 10) 
         
    # fig, ax = plt.subplots()
    # ax.scatter(x_num, e_eps_num, c='r', label='Numerical')
    # ax.plot(x_an, e_eps_an, 'b', label="Analytical")
    # ax.legend()

    e_u_err_max = max(e_u_err_H1)
    e_eps_err_max = max(e_eps_err_L2)
    
    e_w_err_max = max(e_w_err_H1)
    e_kap_err_max = max(e_kap_err_L2)
    
    w_err_max = max(w_err_H1)
    
    return e_u_err_max, e_eps_err_max, e_w_err_max, e_kap_err_max, w_err_max
    


n_h = 5
n_vec = np.array([2**(i+2) for i in range(n_h)])
h_vec = 1./n_vec


e_u_err_deg1 = np.zeros((n_h,))
e_u_err_deg2 = np.zeros((n_h,))
e_u_err_deg3 = np.zeros((n_h,))

e_eps_err_deg1 = np.zeros((n_h,))
e_eps_err_deg2 = np.zeros((n_h,))
e_eps_err_deg3 = np.zeros((n_h,))

e_w_err_deg1 = np.zeros((n_h,))
e_w_err_deg2 = np.zeros((n_h,))
e_w_err_deg3 = np.zeros((n_h,))

e_kap_err_deg1 = np.zeros((n_h,))
e_kap_err_deg2 = np.zeros((n_h,))
e_kap_err_deg3 = np.zeros((n_h,))

e_disp_err_deg1 = np.zeros((n_h,))
e_disp_err_deg2 = np.zeros((n_h,))
e_disp_err_deg3 = np.zeros((n_h,))


for i in range(n_h):
    e_u_err_deg1[i], e_eps_err_deg1[i], e_w_err_deg1[i], e_kap_err_deg1[i], \
        e_disp_err_deg1[i] = compute_err(n_vec[i], 1)
    e_u_err_deg2[i], e_eps_err_deg2[i], e_w_err_deg2[i], e_kap_err_deg2[i], \
        e_disp_err_deg2[i] = compute_err(n_vec[i], 2)
    # e_u_err_deg3[i], e_eps_err_deg3[i], e_w_err_deg3[i], e_kap_err_deg3[i], \
    #     e_disp_err_deg3[i] = compute_err(n_vec[i], 3)

        
       
path_res = "./errors/convergence_vonKarman/"
if save_res:
    np.save(path_res + "h_vec", h_vec)
   
    np.save(path_res + "e_u_err_deg1", e_u_err_deg1)
    np.save(path_res + "e_u_err_deg2", e_u_err_deg2)
    np.save(path_res + "e_u_err_deg3", e_u_err_deg3)

    np.save(path_res + "e_eps_err_deg1", e_eps_err_deg1)
    np.save(path_res + "e_eps_err_deg2", e_eps_err_deg2)
    np.save(path_res + "e_eps_err_deg3", e_eps_err_deg3)
    
    np.save(path_res + "e_w_err_deg1", e_w_err_deg1)
    np.save(path_res + "e_w_err_deg2", e_w_err_deg2)
    np.save(path_res + "e_w_err_deg3", e_w_err_deg3)
    
    np.save(path_res + "e_kap_err_deg1", e_kap_err_deg1)
    np.save(path_res + "e_kap_err_deg2", e_kap_err_deg2)
    np.save(path_res + "e_kap_err_deg3", e_kap_err_deg3)
    
    np.save(path_res + "e_disp_err_deg1", e_disp_err_deg1)
    np.save(path_res + "e_disp_err_deg2", e_disp_err_deg2)
    np.save(path_res + "e_disp_err_deg3", e_disp_err_deg3)


