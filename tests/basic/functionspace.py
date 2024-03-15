import firedrake as fdrk

mesh = fdrk.UnitSquareMesh(1,1, quadrilateral=True)
space = fdrk.FunctionSpace(mesh, "RTCF", 1)

print(space.dim())