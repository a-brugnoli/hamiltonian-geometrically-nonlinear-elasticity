import firedrake as fdrk
import numpy as np
import matplotlib.pyplot as plt
from animation_1d_function import create_function_animation


def bilinear_form(u, v=None):
    """
    Bilinear form for the wave equation, given by:
    <u, v> = (grad(u), grad(v)) dx
    where u and v are both in H^1_0(0,L)

    Parameters:
    u : Trial function
    v : Test function, optional, default is None

    Returns:
    Bilinear form for the wave equation
    """
    if v == None:
        return fdrk.inner(fdrk.grad(u), fdrk.grad(u))*fdrk.dx
    else:
        return fdrk.inner(fdrk.grad(u), fdrk.grad(v))*fdrk.dx
    

def potential_energy(u):
    return 0.5*fdrk.inner(fdrk.grad(u), fdrk.grad(u))*fdrk.dx

def discrete_gradient(var_new, var_old, test_var, grad_H, H):
    """
    Mean value discrete gradient not supported on multiple meshes.
    Perhaps the simplest is to create a tensor product mesh to eliminate the error
    Exploits derivative from firedrake
    Parameters:
    var_new
    H : Hamiltonian functional
    
    """
    var_mid = 0.5*(var_old + var_new)
    var_diff = var_new - var_old

    dH_xmid = grad_H(test_var, var_mid)

    num_coeff = fdrk.assemble(H(x_new) - H(x_old) - grad_H(var_diff, var_mid))
    den_coeff = fdrk.assemble(fdrk.inner(var_diff,var_diff)*fdrk.dx)
    coeff = num_coeff/den_coeff
    dH_discrete = dH_xmid + fdrk.inner(test_var, coeff*var_diff) * fdrk.dx

    return dH_discrete

t_span = np.array([0, 1])
dt = 0.01
simulation_time = t_span[1] - t_span[0]
n_steps = np.round(simulation_time/dt).astype(int)
T_end = n_steps*dt + t_span[0]
t_vec = np.linspace(t_span[0], T_end, n_steps+1)
dt = fdrk.Constant(dt)

n_elements = 10
L = 1
domain = fdrk.IntervalMesh(n_elements, L)
x_coord, = fdrk.SpatialCoordinate(domain)

V = fdrk.FunctionSpace(domain, "CG", 1)

mixedspace = V * V

bc_q = fdrk.DirichletBC(mixedspace.sub(0), fdrk.Constant(0), "on_boundary")
bc_v = fdrk.DirichletBC(mixedspace.sub(1), fdrk.Constant(0), "on_boundary")
bcs = [bc_q, bc_v]

test_q, test_v = fdrk.TestFunctions(mixedspace)

x_old = fdrk.Function(mixedspace)
q_old, v_old = x_old.subfunctions

x_new = fdrk.Function(mixedspace)
q_new, v_new = fdrk.split(x_new)

q0 = fdrk.interpolate(fdrk.sin(fdrk.pi/L*x_coord), mixedspace.sub(0)) 
v0 = fdrk.interpolate(fdrk.Constant(0), mixedspace.sub(1))

q_old.assign(q0)
v_old.assign(v0)

eps = 1e-2
x_guess = fdrk.Function(mixedspace)
x_guess.sub(0).assign(q_old*(1+eps))
x_guess.sub(1).assign(v_old)

x_new.assign(x_guess)

dV_discrete = discrete_gradient(q_new, q_old, test_v, bilinear_form, potential_energy)

q_midpoint = 0.5*(q_new + q_old)
v_midpoint = 0.5*(v_new + v_old)

res_q = fdrk.inner(test_q, q_new - q_old - dt*v_midpoint)*fdrk.dx
res_v = fdrk.inner(test_v, v_new - v_old)*fdrk.dx + dt*dV_discrete

residual = res_q + res_v

energy_old = 0.5 * (fdrk.inner(v_old, v_old) \
                + fdrk.inner(fdrk.grad(q_old), fdrk.grad(q_old)))*fdrk.dx

energy_new = 0.5 * (fdrk.inner(v_new, v_new) \
                + fdrk.inner(fdrk.grad(q_new), fdrk.grad(q_new)))*fdrk.dx

problem = fdrk.NonlinearVariationalProblem(residual, x_new, bcs = bcs)
solver = fdrk.NonlinearVariationalSolver(problem)

energy_vec = np.zeros(n_steps+1)
energy_vec[0] = fdrk.assemble(energy_old)

q_sol = []
v_sol = []
q_sol.append(x_old.copy(deepcopy=True).subfunctions[0])
v_sol.append(x_old.copy(deepcopy=True).subfunctions[1])

for i in range(n_steps):
    solver.solve()

    q_diff = q_new - q_old
    coeff_dis_grad = fdrk.assemble(0.5*(bilinear_form(q_new) - bilinear_form(q_old)) - bilinear_form(q_diff, q_midpoint))

    energy_vec[i+1] = fdrk.assemble(energy_new)
    x_old.assign(x_new)

    x_guess.sub(0).assign(q_old*(1+eps))
    x_guess.sub(1).assign(v_old)

    x_new.assign(x_guess)

    q_sol.append(x_old.copy(deepcopy=True).subfunctions[0])
    v_sol.append(x_old.copy(deepcopy=True).subfunctions[1])


plt.figure()
plt.plot(t_vec, energy_vec)
plt.title("Energy")
anim_q = create_function_animation(q_sol, domain, interval=10, title="Position", display=False)
anim_v = create_function_animation(v_sol, domain, interval=10, title="Velocity", display=False)

plt.show()