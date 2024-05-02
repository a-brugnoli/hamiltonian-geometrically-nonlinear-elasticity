import firedrake as fdrk
from firedrake.petsc import PETSc


def solve_twisting_column(n_elem, pol_degree):


    domain = fdrk.BoxMesh(n_elem, n_elem, n_elem, Lx = 1, Ly = 1, Lz = 6)

    disp_space = fdrk.FunctionSpace(domain, "CG", pol_degree)
    strain_space = fdrk.FunctionSpace(domain, "Regge", pol_degree - 1)
    stress_space = strain_space

    mixed_space = disp_space * strain_space * stress_space
    test_velocity, test_strain, test_stress = fdrk.TestFunctions(mixed_space)

    solution = fdrk.Function(mixed_space)

    displacement = fdrk.Function(disp_space)


