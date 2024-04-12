import firedrake as fdrk

def compute_min_mesh_size(mesh):
    DG0_space = fdrk.FunctionSpace(mesh, 'DG', 0)
    v_DG0 = fdrk.TestFunction(DG0_space)
    diameters = fdrk.CellSize(mesh)
    hvol_form = v_DG0 * diameters * fdrk.dx
    volume_form = v_DG0 * fdrk.dx

    vector_volh = fdrk.assemble(hvol_form).vector().get_local()
    vector_vol = fdrk.assemble(volume_form).vector().get_local()
    vector_h = vector_volh / vector_vol

    return min(vector_h)


def compute_min_max_function(function: fdrk.Function, tuple_min_max):

    previous_min, previous_max = tuple_min_max

    vector = function.vector().get_local()

    present_min = min(vector)
    present_max = max(vector)

    if present_min < previous_min:
        previous_min = present_min

    if present_max > previous_max:
        previous_max = present_max

    return (previous_min, previous_max)


def compute_min_max_mesh(mesh: fdrk.MeshGeometry, *previous_list_min_max):

    dim = mesh.geometric_dimension()

    assert len(previous_list_min_max)==dim

    list_tuple = []

    for i in range(dim):
        previous_min, previous_max = previous_list_min_max[i]
        coordinates = mesh.coordinates.dat.data[:, i]

        present_min = min(coordinates)
        present_max = max(coordinates)

        if present_min < previous_min:
            previous_min = present_min

        if present_max > previous_max:
            previous_max = present_max

        list_tuple.append((present_min, present_max))

    return list_tuple