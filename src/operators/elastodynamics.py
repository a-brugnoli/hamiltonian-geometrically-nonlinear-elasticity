import firedrake as fdrk
from system_operators import SystemOperators
from src.problems.problem import Problem

class Elastodynamics(SystemOperators):

    def __init__(problem: Problem, pol_degree: int):
        super().__init__()


    def _set_space(self):
        DG_tensorspace = fdrk.TensorFunctionSpace(mesh, "DG", pol_degree-1)

        CG_vectorspace = fdrk.VectorFunctionSpace(mesh, "CG", pol_degree)
        DG_symtensorspace = fdrk.TensorFunctionSpace(mesh, "DG", pol_degree-1, symmetry=True)

        space_energy = CG_vectorspace * DG_symtensorspace
