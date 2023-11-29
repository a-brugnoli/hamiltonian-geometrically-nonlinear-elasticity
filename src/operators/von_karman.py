import firedrake as fdrk
from operators.abstract_operators import HamiltonianOperators
from src.problems.problem import Problem

class HamiltonianVonKarmanPlate(HamiltonianOperators):

    def __init__(self, problem : Problem, pol_degree : int):
        super().__init__(problem, pol_degree)

    def _set_space(self):

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", self.pol_degree+1)
        DG_symtensorspace = fdrk.TensorFunctionSpace(self.domain, "DG", self.pol_degree, symmetry= True)

        DG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "DG", self.pol_degree)

        CG_scalarspace = fdrk.FunctionSpace(self.domain, "CG", self.pol_degree)
        HHJ_symtensorspace = fdrk.FunctionSpace(self.domain, "HHJ", self.pol_degree-1)

        self.space_defgradient = DG_vectorspace