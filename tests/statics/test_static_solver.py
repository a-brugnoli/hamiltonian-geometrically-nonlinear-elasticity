import firedrake as fdrk
# from src.preprocessing.static_parser import *

n_el = 5
nx, ny, nz = n_el, n_el, n_el
pol_degree = 1

domain = fdrk.UnitCubeMesh(nx, ny, nz)
dim = domain.geometric_dimension()

x, y, z = fdrk.SpatialCoordinate(domain)

CG_vectorspace = fdrk.VectorFunctionSpace(domain, "CG", pol_degree)
NED_vectorspace = fdrk.VectorFunctionSpace(domain, "N1curl", pol_degree, dim=2)
# The different components are piled columns wise : size (2, 3). Every row is a Nedelec element


test_def_gradient = fdrk.TestFunction(NED_vectorspace)
trial_def_gradient = fdrk.TrialFunction(NED_vectorspace)

displacement_ = fdrk.Function(CG_vectorspace)
def_gradient_ = fdrk.Function(NED_vectorspace)

print(def_gradient_.ufl_shape)
