import gmsh
import firedrake as fdrk
import matplotlib.pyplot as plt
import sys


def create_cook_membrane(mesh_size, quad=False):
    # Before using any functions in the Python API, Gmsh must be initialized:
    gmsh.initialize()

    gmsh.model.add("cook_membrane")

    lc = mesh_size
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(48, 44, 0, lc)
    p3 = gmsh.model.geo.addPoint(48, 60, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, 44, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p4)
    l2 = gmsh.model.geo.addLine(p4, p3)
    l3 = gmsh.model.geo.addLine(p3, p2)
    l4 = gmsh.model.geo.addLine(p2, p1)

    loop1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4])

    surface1 = gmsh.model.geo.addPlaneSurface([loop1])

    gmsh.model.addPhysicalGroup(1, [l1], 1, name="Clamp")
    gmsh.model.addPhysicalGroup(1, [l2], 2, name="Free1")
    gmsh.model.addPhysicalGroup(1, [l3], 3, name="Traction")
    gmsh.model.addPhysicalGroup(1, [l4], 4, name="Free2")

    # gmsh.model.addPhysicalGroup(2, [surface1], name="Surface")

    # Remember that by default, if physical groups are defined, Gmsh will export in
    # the output mesh file only those elements that belong to at least one physical
    # group. To force Gmsh to save all elements, you can use

    if quad:
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Quadrilateral elements
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.SaveAll", 1)

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)

    # ... and save it to disk
    gmsh.write("cook_membrane.msh")

    # To visualize the model we can run the graphical user interface with
    # `gmsh.fltk.run()'. Here we run it only if "-nopopup" is not provided in the
    # command line arguments:
    
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    # This should be called when you are done using the Gmsh Python API:

    gmsh.finalize()


if __name__ == "__main__":
    mesh_size = 1
    create_cook_membrane(mesh_size)


    domain = fdrk.Mesh('cook_membrane.msh')

    fig, axes = plt.subplots()
    fdrk.triplot(domain, axes=axes)
    axes.legend()
    plt.show()