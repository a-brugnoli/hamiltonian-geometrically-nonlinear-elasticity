import firedrake as fdrk
from src.solvers.dynamics.hamiltonian_neo_hookean import energy_density_neo_hookean, second_piola_definition

E = 1
nu = 0.3

alpha = 0.05
beta = 0.07
gamma = 0.04
delta = 0.03
domain = fdrk.BoxMesh(10, 10, 10, Lx=1, Ly=1, Lz=1)
DG_tensorspace = fdrk.TensorFunctionSpace(domain, "CG", 2, symmetry=True)
C = fdrk.as_tensor([[1+alpha, 0, delta], 
                    [0, 1+beta, 0], 
                    [delta, 0, 1+gamma]])

# C = fdrk.interpolate(C_exp, DG_tensorspace)

psi = energy_density_neo_hookean(C, young_modulus=E, poisson_ratio=nu)

S_automatic = 2 * fdrk.diff(psi, fdrk.variable(C))

print(S_automatic)
stress_function_auto = fdrk.interpolate(S_automatic, DG_tensorspace)

S_manual = second_piola_definition(C, E, nu)
stress_function_manual = fdrk.interpolate(S_manual, DG_tensorspace)

print(f"Error {fdrk.norm(stress_function_manual - stress_function_auto)}")
