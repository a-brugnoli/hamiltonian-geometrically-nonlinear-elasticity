import matplotlib.pyplot as plt
from firedrake import *
length = 1
width = 0.2
mesh = RectangleMesh(40, 20, length, width)

V = VectorFunctionSpace(mesh, "Lagrange", 1)

bc = DirichletBC(V, Constant([0, 0]), 1)

rho = Constant(0.01)
g = Constant(1)
f = as_vector([0, -rho*g])
mu = Constant(1)
lambda_ = Constant(0.25)
Id = Identity(mesh.geometric_dimension()) # 2x2 Identity tensor


def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def sigma(u):
    return lambda_*div(u)*Id + 2*mu*epsilon(u)

u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx

uh = Function(V)
solve(a == L, uh, bcs=bc, solver_parameters={"ksp_monitor": None})

displaced_coordinates = interpolate(SpatialCoordinate(mesh) + uh, V)
displaced_mesh = Mesh(displaced_coordinates)

# NBVAL_IGNORE_OUTPUT
fig, axes = plt.subplots()
triplot(displaced_mesh, axes=axes)
axes.set_aspect("equal");

plt.show()