import firedrake as fdrk

def first_piola_definition(grad_disp):
        mu = 80.194  #N/mm^2
        lamda = 400889.8 #N/mm^2

        def_grad = fdrk.Identity(2) + grad_disp
        inv_F_transpose = fdrk.inv(def_grad.T)
        J = fdrk.det(def_grad)
        return mu*(def_grad - inv_F_transpose) + lamda * fdrk.ln(J) * inv_F_transpose


domain = fdrk.UnitSquareMesh(5, 5)
coordinates_mesh = fdrk.SpatialCoordinate(domain)
x, y = coordinates_mesh

# exact_displacement = fdrk.as_vector([0.5*y**3 + 0.5*fdrk.sin(0.5 * fdrk.pi * y), fdrk.Constant(0)])

exact_displacement = fdrk.as_vector([0.5*y**3 + 0.5*fdrk.sin(0.5 * fdrk.pi * y), 0])

normal_versor = fdrk.FacetNormal(domain)
normal_displacement = fdrk.dot(exact_displacement, normal_versor)

exact_grad_disp = fdrk.grad(exact_displacement)
exact_first_piola = first_piola_definition(exact_grad_disp)

pol_degree = 1
BDM_space = fdrk.FunctionSpace(domain, "BDM", pol_degree)
BDM_vectorspace = fdrk.VectorFunctionSpace(domain, "BDM", pol_degree)


column = exact_first_piola[:, 0]

bc_BDM = fdrk.DirichletBC(BDM_vectorspace.sub(0), column, 1)

