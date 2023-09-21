import gmsh

mesh_size=0.1
# Initialize Gmsh
gmsh.initialize()

# Define the geometry
center_x = 0.5
center_y = 0.5
radius = 0.3
size_x = 1
side_y = 1
square = gmsh.model.occ.addRectangle(0, 0, 0, size_x, side_y)
circle = gmsh.model.occ.addDisk(center_x, center_y, 0, rx=radius, ry=radius)

resulting_geometry = gmsh.model.occ.cut([(2, square)], [(2, circle)])

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
gmsh.model.mesh.generate(2)

# Save the mesh to a file (optional)
gmsh.write("square_hole.msh")

# Finalize Gmsh
gmsh.finalize()
