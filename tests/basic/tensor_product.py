import firedrake as fdrk

nel_x = 1 
nel_y = 1
L=1

base_mesh = fdrk.IntervalMesh(nel_x, length_or_left=L)
mesh = fdrk.ExtrudedMesh(base_mesh, nel_y, layer_height=int(L/nel_y))

CG_int = fdrk.FiniteElement("CG", fdrk.interval, 1)
Her_int = fdrk.FiniteElement("Hermite", fdrk.interval, 3)
tensor_CG = fdrk.TensorProductElement(CG_int, CG_int)
tensor_Her = fdrk.TensorProductElement(Her_int, Her_int)   
V = fdrk.FunctionSpace(mesh, tensor_CG)

v = fdrk.TestFunction(V)
w = fdrk.TrialFunction(V)

mass_form = v*w*fdrk.dx
M = fdrk.assemble(mass_form)

# x, y = fdrk.SpatialCoordinate(mesh)
# u = fdrk.project(x*y, V)

