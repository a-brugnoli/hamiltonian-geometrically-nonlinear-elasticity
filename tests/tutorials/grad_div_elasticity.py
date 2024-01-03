import firedrake as fdrk

mesh = fdrk.UnitSquareMesh(3, 3)
pol_degree = 1
dim = 7
CG_vectorspace= fdrk.VectorFunctionSpace(mesh, "CG", pol_degree, dim=dim)
u = fdrk.Function(CG_vectorspace)
print(u.ufl_shape)
grad_u = fdrk.grad(u)
print(grad_u.ufl_shape)
BDM_vectorspace = fdrk.VectorFunctionSpace(mesh, "BDM", pol_degree, dim=dim)
first_piola = fdrk.Function(BDM_vectorspace)
print(first_piola.ufl_shape)
div_first_piola = fdrk.div(first_piola)
print(div_first_piola.ufl_shape)