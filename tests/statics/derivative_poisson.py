import firedrake as fdrk
import matplotlib.pyplot as plt

# Create mesh (unit square)
n = 32
mesh = fdrk.UnitSquareMesh(n, n)

# Define function space (continuous Galerkin)
V = fdrk.FunctionSpace(mesh, "CG", 1)

# Define the trial function (the solution we're solving for)
u = fdrk.Function(V)
# Define the test function (the variation)
v = fdrk.TestFunction(V)

# Define source term f(x,y) = 8π²sin(2πx)sin(2πy)
x, y = fdrk.SpatialCoordinate(mesh)
f = 8*fdrk.pi*fdrk.pi*fdrk.sin(2*fdrk.pi*x)*fdrk.sin(2*fdrk.pi*y)

# Define the Dirichlet energy functional:
# E(u) = ∫(1/2|∇u|² - fu)dx
energy = 0.5 * fdrk.dot(fdrk.grad(u), fdrk.grad(u))*fdrk.dx - f*u*fdrk.dx

# Take variational derivative of the energy
# δE/δu = 0 gives us the Euler-Lagrange equation
dE = fdrk.derivative(energy, u, v)

# Define Dirichlet boundary condition (u = 0 on boundary)
bc = fdrk.DirichletBC(V, 0.0, "on_boundary")

# Solve the variational problem δE/δu = 0
fdrk.solve(dE == 0, u, bcs=bc)

# The exact solution for comparison
u_exact = fdrk.Function(V)
u_exact.interpolate(fdrk.sin(2*fdrk.pi*x)*fdrk.sin(2*fdrk.pi*y))

fdrk.trisurf(u)
fdrk.trisurf(u_exact)

plt.show()