import matplotlib.pyplot as plt
import firedrake as fdrk
from petsc4py import PETSc
import scipy.sparse as sp
from src.postprocessing.options import configure_matplotib
configure_matplotib()

def create_1d_laplacian(n):
    """
    Create a PETSc matrix representing the 1D Laplacian operator using finite differences.
    
    For a 1D problem with n grid points, the discretized Laplacian using second-order 
    central differences is a tridiagonal matrix with:
    - 2 on the main diagonal
    - -1 on the first sub- and super-diagonals
    
    Parameters:
    -----------
    n : int
        Number of grid points
        
    Returns:
    --------
    A : PETSc.Mat
        PETSc matrix representing the 1D Laplacian
    """
    # Create a PETSc matrix
    A = PETSc.Mat().create()
    A.setSizes([n, n])
    A.setType('aij')  # Sparse AIJ format
    
    # Preallocate memory (3 non-zeros per row - tridiagonal)
    A.setPreallocationNNZ(3)
    
    # Assemble the matrix
    A.setUp()
    
    # Set the values for the Laplacian operator: -u_{i-1} + 2u_i - u_{i+1}
    for i in range(n):
        # Main diagonal: 2
        A.setValue(i, i, 2.0)
        
        # Sub-diagonal: -1
        if i > 0:
            A.setValue(i, i-1, -1.0)
            
        # Super-diagonal: -1
        if i < n-1:
            A.setValue(i, i+1, -1.0)
    
    # Assemble the matrix
    A.assemblyBegin()
    A.assemblyEnd()
    
    return A


def create_firedrake_mass_matrix(n_elements, family, length= 1, element_degree=1, dim=1, variant="equispaced"):
    """
    Create a mass matrix using Firedrake finite element library.
    
    Parameters:
    -----------
    n_elements : int
        Number of elements in each dimension
    element_degree : int, optional
        Polynomial degree for the finite element (default: 1)
    dim : int, optional
        Dimension of the problem (1, 2, or 3) (default: 1)
        
    Returns:
    --------
    M : PETSc.Mat
        PETSc matrix representing the mass matrix
    """
    # Create mesh based on dimension
    if dim == 1:
        mesh = fdrk.IntervalMesh(n_elements, length)
    elif dim == 2:
        mesh = fdrk.SquareMesh(n_elements, n_elements, length)
    elif dim == 3:
        mesh = fdrk.CubeMesh(n_elements, n_elements, n_elements, length)
    else:
        raise ValueError("Dimension must be 1, 2, or 3")
    
    # Create function space with Lagrange elements of specified degree
    cell = mesh.ufl_cell()
    fe_space = fdrk.FiniteElement(family, cell, element_degree, variant=variant)
    V = fdrk.FunctionSpace(mesh, fe_space)

    # Define trial and test functions
    u = fdrk.TrialFunction(V)
    v = fdrk.TestFunction(V)
    
    # Define the mass matrix form
    mass_form = u * v * fdrk.dx
    
    # Assemble the mass matrix
    M = fdrk.assemble(mass_form).M.handle
    
    return M


def plot_sparsity_pattern(A):
    """
    Plot the sparsity pattern of a PETSc matrix using matplotlib.
    Properly handling the CSR format.
    """
    # Get matrix dimensions
    m, n = A.getSize()
    
    # Get CSR representation
    csr = A.getValuesCSR()
    indptr, indices, data = csr
    
    # Convert to scipy sparse matrix
    scipy_mat = sp.csr_matrix((data, indices, indptr), shape=(m, n))
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.spy(scipy_mat, markersize=10)
    plt.title(f'Sparsity Pattern of matrix ({m}×{n})')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_el = 1

    # # 1D Laplacian using finite differences
    # print("Creating 1D Laplacian matrix with finite differences...")
    # laplacian_fd = create_1d_laplacian(n_size)
    # nnz_laplacian = plot_sparsity_pattern(laplacian_fd)
    # print(f"1D Laplacian size: {laplacian_fd.getSize()}, nonzeros: {nnz_laplacian}")
    
    # 1D mass matrix using Firedrake (DG elements)
    degree = 4
    print("\nCreating 1D mass matrix with Firedrake ...")

    # Variant "equispaced" uses equidistant nodes
    # Variant "spectral" uses Legendre points
    length = 1
    mass_petsc = create_firedrake_mass_matrix(n_elements=n_el, family="DG", length=length,
                                              element_degree=degree,
                                              variant="spectral")

    mass_scipy = sp.csr_matrix(mass_petsc.getValuesCSR()[::-1])
    mass_dense = mass_scipy.todense()
    print('Mass matrix in scipy format: \n', mass_dense)
    nnz_mass= plot_sparsity_pattern(mass_petsc)
    