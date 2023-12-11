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

mu = 80.194  #N/mm2
lmbda = 400889.8 #N/mm

def first_piola_expression(def_grad):
    inv_F_transpose = fdrk.inv(def_grad).T
    return mu*(def_grad - inv_F_transpose) + 4 * lmbda * fdrk.ln(fdrk.det(def_grad)) * inv_F_transpose
    # return (def_grad) 


def derivative_first_piola(tensor, def_grad):
    invF = fdrk.inv(def_grad)
    inv_Ftr = fdrk.inv(def_grad).T

    return mu * tensor + (mu - 4 * lmbda * fdrk.ln(fdrk.det(def_grad))) * fdrk.dot(inv_Ftr, fdrk.dot(tensor.T, inv_Ftr)) \
            + 4 * lmbda * fdrk.tr(fdrk.dot(invF, tensor)) * inv_Ftr


n_el = 20
nx, ny = n_el, n_el
pol_degree = 2

length_side = 0.01
domain = fdrk.RectangleMesh(nx, ny, Lx = length_side, Ly = length_side)
dim = domain.geometric_dimension()

coordinates_mesh = fdrk.SpatialCoordinate(domain)
x, y = coordinates_mesh

CG_vectorspace = fdrk.VectorFunctionSpace(domain, "CG", pol_degree)
NED_vectorspace = fdrk.VectorFunctionSpace(domain, "N2curl", pol_degree-1) # Every row is a Nedelec

mixed_space_grad = CG_vectorspace * NED_vectorspace * NED_vectorspace
test_disp, test_first_piola, test_def_grad = fdrk.TestFunctions(mixed_space_grad)


solution_ = fdrk.Function(mixed_space_grad)

trial_disp, trial_first_piola, trial_def_grad = fdrk.split(solution_)

ampl= 1
force = fdrk.as_vector([0, -ampl])*fdrk.conditional(fdrk.le(x, length_side/2), 1, 0) 

res_equilibrium = fdrk.inner(fdrk.grad(test_disp), trial_first_piola) * fdrk.dx - fdrk.inner(test_disp, force) * fdrk.ds(4)
res_stress = fdrk.inner(test_first_piola, trial_first_piola - first_piola_expression(trial_def_grad)) * fdrk.dx
res_def_grad = fdrk.inner(test_def_grad, trial_def_grad - (fdrk.Identity(dim) + fdrk.grad(trial_disp)))*fdrk.dx


res = res_equilibrium + res_stress + res_def_grad


displacement, first_piola, def_gradient = solution_.subfunctions

hessian_potential = fdrk.diff(first_piola_expression(def_gradient), def_gradient)


bcs_bottom = fdrk.DirichletBC(mixed_space_grad.sub(0).sub(1), fdrk.Constant(0), 3)
bcs_top = fdrk.DirichletBC(mixed_space_grad.sub(0).sub(0), fdrk.Constant(0), 4)
bcs_symmetry = fdrk.DirichletBC(mixed_space_grad.sub(0).sub(0), fdrk.Constant(0), 1)
bcs = [bcs_bottom, bcs_top, bcs_symmetry]

problem = fdrk.NonlinearVariationalProblem(res, solution_, bcs = bcs)

solver = fdrk.NonlinearVariationalSolver(problem, solver_parameters={"ksp_type": "gmres"})

solver.solve()


int_coordinates = fdrk.Mesh(fdrk.interpolate(coordinates_mesh, CG_vectorspace))
int_displaced_coordinates = fdrk.Mesh(fdrk.interpolate(coordinates_mesh  + displacement, CG_vectorspace))

fig, axes = plt.subplots()
# fdrk.triplot(int_coordinates, axes=axes)
fdrk.triplot(int_displaced_coordinates, axes=axes)

plt.show()