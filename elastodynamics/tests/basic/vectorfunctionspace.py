import firedrake as fdrk

mesh = fdrk.UnitSquareMesh(5,5)
vector_CG = fdrk.VectorFunctionSpace(mesh, "CG", 1)

vector = fdrk.Function(vector_CG)