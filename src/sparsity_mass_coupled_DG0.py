import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib.pyplot import spy
import scipy.sparse as sp
from src.postprocessing.options import configure_matplotib
configure_matplotib()

E = 70e9  # Young's modulus
nu = 0.3  # Poisson's ratio

def compliance_tensor(stress_tensor):
        dim = stress_tensor.ufl_shape[0]
        # Compliance tensor for generic dimensions
        strain_tensor = 1 /E * ((1+nu)*stress_tensor \
                - nu * fd.Identity(dim) * fd.tr(stress_tensor))
        return strain_tensor

# Create a unit box mesh of hexahedra
n = 1  # Number of elements in each direction
mesh = fd.UnitCubeMesh(n, n, n, hexahedral=False)

# Define a tensor-valued symmetric DG_0 function space
# For a symmetric tensor in 3D, we need a space with 6 components
# (xx, xy, xz, yy, yz, zz)
element = fd.TensorElement("DG", mesh.ufl_cell(), 0, symmetry=True)
V = fd.FunctionSpace(mesh, element)
n_dofs_V = V.dim()
print(f"Number of DOFs: {n_dofs_V}")


# Define the test and trial functions
tau = fd.TestFunction(V)
sigma = fd.TrialFunction(V)

a = fd.inner(tau, compliance_tensor(sigma)) * fd.dx

# Assemble the mass matrix
A = fd.assemble(a).M.handle

# Convert the PETSc matrix to scipy sparse format for visualization
A_scipy = sp.csr_matrix(A.getValuesCSR()[::-1])

# Plot the sparsity pattern
plt.figure(figsize=(10, 8))
spy(A_scipy, markersize=15)
plt.title("Sparsity pattern of the mass matrix for the compliance mass matrix")
plt.tight_layout()
plt.savefig("compliance_mass_matrix_sparsity.pdf", dpi=300)

plt.show()

# mixed_space = V * V

# # Print information about the function space

# # Define the test and trial functions
# tau_1, tau_2 = fd.TestFunctions(mixed_space)
# sigma_1, sigma_2 = fd.TrialFunctions(mixed_space)

# # Define the mass matrix form
# a_mixed = fd.inner(tau_1 + tau_2, compliance_tensor(sigma_1 + sigma_2)) * fd.dx

# # Assemble the mass matrix
# A_mixed = fd.assemble(a_mixed).M.handle

# # Convert the PETSc matrix to scipy sparse format for visualization
# A_mixed_scipy = sp.csr_matrix(A_mixed.getValuesCSR()[::-1])

# # Plot the sparsity pattern
# plt.figure(figsize=(10, 8))
# spy(A_mixed_scipy, markersize=5)
# plt.title("Sparsity pattern of the mixed mass matrix for coupled symmetric tensors")
# plt.tight_layout()
# plt.savefig("mixed_compliance_mass_matrix_sparsity.pdf", dpi=300)

# A_mixed_block_12 = A_mixed_scipy[:n_dofs_V, n_dofs_V:]

# # Plot the sparsity pattern
# plt.figure(figsize=(10, 8))
# spy(A_mixed_block_12, markersize=5)
# plt.title("Block 12 of the mass matrix")
# plt.tight_layout()
# plt.savefig("block_12_compliance_mass_matrix_sparsity.pdf", dpi=300)
# plt.show()