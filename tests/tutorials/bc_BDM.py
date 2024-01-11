import firedrake as fdrk

domain = fdrk.UnitSquareMesh(5, 5)
coordinates_mesh = fdrk.SpatialCoordinate(domain)
x, y = coordinates_mesh

exact_displacement = fdrk.as_vector([0.5*y**3 + 0.5*fdrk.sin(0.5 * fdrk.pi * y), fdrk.Constant(0)])
normal_versor = fdrk.FacetNormal(domain)
normal_displacement = fdrk.dot(exact_displacement, normal_versor)

BDM_space = fdrk.FunctionSpace(domain, "BDM", 1)

bc_BDM = fdrk.DirichletBC(BDM_space, exact_displacement, 1)
