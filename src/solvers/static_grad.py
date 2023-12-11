import firedrake as fdrk
from src.problems.problem import Problem

class StaticSolverGrad:

    def __init__(self, problem:Problem, pol_degree:int):

        self.domain = problem.domain

        NED_space = fdrk.FunctionSpace(self.domain, "N1curl", pol_degree)

        CG_vectorspace = fdrk.VectorFunctionSpace(self.domain, "CG", pol_degree)



        