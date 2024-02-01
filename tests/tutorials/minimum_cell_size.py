from firedrake import *
import matplotlib.pyplot as plt
import gmsh

def unstructured_unit_square(lc):
    gmsh.initialize()

    # lc = 0.1  # Characteristic mesh size
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    ll = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    pl = gmsh.model.geo.addPlaneSurface([ll])

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)

    gmsh.model.mesh.setRecombine(2, pl)

    gmsh.model.geo.mesh.setTransfiniteCurve(l1, int(1/lc))
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, int(1/lc))
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, int(1/lc))
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, int(1/lc))

    gmsh.model.geo.mesh.setTransfiniteSurface(1, "Left", [1, 2, 3, 4])

    gmsh.option.setNumber("Mesh.Smoothing", 100)

    gmsh.model.mesh.generate(2)  # 2D meshing

    gmsh.write("unit_square.msh")

    gmsh.finalize()


# unstructured_unit_square(0.5)
# # Create mesh and function space
# mesh = Mesh('unit_square.msh')

mesh = UnitSquareMesh(10,10,quadrilateral=True)
triplot(mesh)
plt.show()


V = FunctionSpace(mesh, 'DG', 0)

# Define a function on the function space
v = TestFunction(V)

# Compute the diameter of each cell
diameters = CellSize(mesh)
diameters
hvol_form = v * diameters * dx
volume_form = v * dx

vector_volh = assemble(hvol_form).vector().get_local()
vector_vol = assemble(volume_form).vector().get_local()

vector_h = vector_volh / vector_vol

print(min(vector_h))