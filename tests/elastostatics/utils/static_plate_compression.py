import firedrake as fdrk
import matplotlib.pyplot as plt

# from src.preprocessing.static_parser import *

# Neo Hookean Potentials
# I_1, I_2, I_3 are the principal invariants of the Cauchy Green deformation tensor C = F^T F
# W_1 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/2 * (I_3^(1/2) - 1)^2
# W_2 = mu/2 * (I_1 - 3) - mu/2 * ln I_3 + kappa/8 *ln(I_3)^2

# First Piola stress tensor
# P_1 = mu (F - F^{-T}) + kappa (J^2 - J) F^{-T}
# P_2 = mu (F - F^{-T}) + kappa ln(J) F^{-T}

mu = 80  #N/mm^2
kappa = 400889 #N/mm^2
length_side = 10 #mm
f = 10 #N/mm

# mu = 80 * 10**(6) #N/m^2
# kappa = 400889 * 10**(6) #N/m^2
# length_side = 0.01  #m
# f = 1000 #N/m

factor = 1

n_el = 30
nx, ny = n_el, n_el
pol_degree = 2

domain = fdrk.RectangleMesh(nx, ny, Lx = length_side, Ly = length_side)
dim = domain.geometric_dimension()
normal = fdrk.FacetNormal(domain)

def first_piola_definition(grad_disp):
    def_grad = fdrk.Identity(dim) + grad_disp
    inv_F_transpose = fdrk.inv(def_grad).T
    return mu*(def_grad - inv_F_transpose) + kappa * fdrk.ln(fdrk.det(def_grad)) * inv_F_transpose


def derivative_first_piola(tensor, grad_disp):
    def_grad = fdrk.Identity(dim) + grad_disp
    invF = fdrk.inv(def_grad)
    inv_Ftr = fdrk.inv(def_grad).T

    return mu * tensor + (mu - kappa * fdrk.ln(fdrk.det(def_grad))) * fdrk.dot(inv_Ftr, fdrk.dot(tensor.T, inv_Ftr)) \
            + kappa * fdrk.tr(fdrk.dot(invF, tensor)) * inv_Ftr


coordinates_mesh = fdrk.SpatialCoordinate(domain)
x, y = coordinates_mesh

CG_vectorspace = fdrk.VectorFunctionSpace(domain, "CG", pol_degree)
NED1_vectorspace = fdrk.VectorFunctionSpace(domain, "N1curl", pol_degree) # Every row is a Nedelec
NED2_vectorspace = fdrk.VectorFunctionSpace(domain, "N2curl", pol_degree-1) # Every row is a Nedelec

BDM_vectorspace = fdrk.VectorFunctionSpace(domain, "BDM", pol_degree) # Every row is a BDM

disp_space = CG_vectorspace
stress_space = NED2_vectorspace
strain_space = NED2_vectorspace

mixed_space_grad = disp_space * stress_space * strain_space
test_disp, test_first_piola, test_grad_disp = fdrk.TestFunctions(mixed_space_grad)


solution_ = fdrk.Function(mixed_space_grad)


disp__, first_piola__, grad_disp__ = fdrk.split(solution_)

force = fdrk.as_vector([0, -factor*f])*fdrk.conditional(fdrk.le(x, length_side/2), 1, 0) 

res_equilibrium = fdrk.inner(fdrk.grad(test_disp), first_piola__) * fdrk.dx - fdrk.inner(test_disp, force) * fdrk.ds(4)
res_stress = fdrk.inner(test_first_piola, first_piola__ - first_piola_definition(grad_disp__)) * fdrk.dx
res_def_grad = fdrk.inner(test_grad_disp, grad_disp__ - fdrk.grad(disp__))*fdrk.dx
res = res_equilibrium + res_stress + res_def_grad

displacement, first_piola, grad_disp = solution_.subfunctions

# piola_guess = fdrk.as_vector([[0, 0],
#                               [0, -factor*f*fdrk.conditional(fdrk.le(x, length_side/2), 1, 0)]])
# first_piola.interpolate(piola_guess)

trial_mixed = fdrk.TrialFunction(mixed_space_grad)
trial_disp, trial_first_piola, trial_grad_disp = fdrk.split(trial_mixed)

# H is the gradient of the displacement
D_res_u_DP = fdrk.inner(fdrk.grad(test_disp), trial_first_piola) * fdrk.dx 
D_res_P_DP = fdrk.inner(test_first_piola, trial_first_piola) * fdrk.dx
D_res_P_DH = fdrk.inner(test_first_piola, - derivative_first_piola(trial_grad_disp, grad_disp__)) * fdrk.dx
D_res_H_DH = fdrk.inner(test_grad_disp, trial_grad_disp)*fdrk.dx
D_res_H_Du = fdrk.inner(test_grad_disp, - fdrk.grad(trial_disp))*fdrk.dx

Jacobian = D_res_u_DP \
         + D_res_P_DP \
         + D_res_P_DH \
         + D_res_H_DH \
         + D_res_H_Du



bcs_bottom = fdrk.DirichletBC(mixed_space_grad.sub(0).sub(1), fdrk.Constant(0), 3)
bcs_top = fdrk.DirichletBC(mixed_space_grad.sub(0).sub(0), fdrk.Constant(0), 4)
bcs_symmetry = fdrk.DirichletBC(mixed_space_grad.sub(0).sub(0), fdrk.Constant(0), 1)
bcs = [bcs_bottom, bcs_top, bcs_symmetry]

problem = fdrk.NonlinearVariationalProblem(res, solution_, bcs = bcs, J=Jacobian)

solver_parameters={'snes_monitor': None,
                   'snes_view': None,
                   'snes_converged_reason': None,
                   'ksp_monitor_true_residual': None,
                   'ksp_converged_reason': None,
                   'ksp_view': None}

test_parameters = {"snes_test_jacobian": 1e-4, "snes_test_jacobian_view": None}

linear_parameters = {'ksp_monitor_true_residual': None,
                    'ksp_converged_reason': None,
                    'ksp_view': None,
                    'ksp_type': 'preonly', 'pc_type': 'lu'}

solver = fdrk.NonlinearVariationalSolver(problem, solver_parameters={})

solver.solve()


int_coordinates = fdrk.Mesh(fdrk.interpolate(coordinates_mesh, CG_vectorspace))
int_displaced_coordinates = fdrk.Mesh(fdrk.interpolate(coordinates_mesh  + displacement, CG_vectorspace))

fig, axes = plt.subplots()
# fdrk.triplot(int_coordinates, axes=axes)
fdrk.triplot(int_displaced_coordinates, axes=axes)

plt.show()