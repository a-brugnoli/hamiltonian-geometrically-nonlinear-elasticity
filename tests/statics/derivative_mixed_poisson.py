import firedrake as fdrk
import matplotlib.pyplot as plt
# Create mesh
n = 32
mesh = fdrk.UnitSquareMesh(n, n)
dx = fdrk.dx(mesh)

# Define mixed function space
# P2 elements for u and P1 elements for p (flux)
V = fdrk.FunctionSpace(mesh, "CG", 1)  # for u
W = fdrk.FunctionSpace(mesh, "N1curl", 1)  # for sigma
mixed_space = V * W

# Define trial and test functions
solution = fdrk.Function(mixed_space)
u, sigma = fdrk.split(solution)  # Trial functions
v, tau = fdrk.TestFunctions(mixed_space)  # Test functions

# Define exact solution term u = sin(πx)sin(πy)
x, y = fdrk.SpatialCoordinate(mesh)

u_ex_exp = fdrk.sin(fdrk.pi*x)*fdrk.sin(fdrk.pi*y)
sigma_ex_exp = fdrk.grad(u_ex_exp)
f = -fdrk.div(fdrk.grad(u_ex_exp))

# Define the energy functional for the mixed formulation
# E(u,sigma) = ∫(1/2|sigma|² + sigma·∇u - fu)dx
energy = (0.5 * fdrk.dot(sigma, sigma) * dx -  # 1/2|sigma|²
         fdrk.dot(sigma, fdrk.grad(u)) * dx +  # sigma·∇u
         f * u * dx)  # -fu

# Take variational derivative of the energy
# This gives us the mixed weak form
dE = fdrk.derivative(energy, solution)

# Define Dirichlet boundary condition for u
bc = fdrk.DirichletBC(mixed_space.sub(0), 0.0, "on_boundary")

# Solve the mixed problem
fdrk.solve(dE == 0, solution, bcs=bc)

# Extract solutions
u_sol, sigma_sol = solution.split()

# Compute exact solution for comparison
u_ex = fdrk.Function(V)
u_ex.interpolate(u_ex_exp)
sigma_ex = fdrk.Function(W)
sigma_ex.interpolate(sigma_ex_exp)

# Compute L2 errors
u_error = fdrk.errornorm(u_sol, u_ex, norm_type="L2", degree_rise=3)
sigma_error = fdrk.errornorm(sigma_sol, sigma_ex, norm_type="L2", degree_rise=3)

print(f"L2 error in u: {u_error}")
print(f"L2 error in p: {sigma_error}")

# Plot solutions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fdrk.trisurf(u_sol, axes=ax)
ax.set_title("u numerical")

fdrk.trisurf(u_ex)

fdrk.quiver(sigma_sol)
fdrk.quiver(sigma_ex)

plt.show()
# Compute and print the energy at the solution
energy_value = fdrk.assemble(energy)
print(f"Energy at solution: {energy_value}")