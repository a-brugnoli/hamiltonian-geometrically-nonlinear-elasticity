import firedrake as fdrk
import numpy as np

def compute_time_step(domain : fdrk.MeshGeometry, displacement : fdrk.Function, \
                      parameters, coeff_cfl = 0.8):
    
    ndim = domain.topological_dimension()
    density = parameters["rho"]
    mu = parameters["mu"]
    kappa = parameters["kappa"]

    coordinates_mesh = fdrk.SpatialCoordinate(domain)
    displaced_domain = fdrk.Mesh(fdrk.interpolate(coordinates_mesh + displacement,\
                                            displacement.function_space()))

    DG0_space = fdrk.FunctionSpace(domain, 'DG', 0)
    vector_J = fdrk.assemble(fdrk.interpolate(fdrk.det(fdrk.Identity(ndim)\
                             + fdrk.grad(displacement)), DG0_space)).vector().get_local()

    DG0_space_displaced = fdrk.FunctionSpace(displaced_domain, 'DG', 0)
    diameters = fdrk.CellSize(displaced_domain)
    vector_h = fdrk.assemble(fdrk.interpolate(diameters, DG0_space_displaced)).vector().get_local()

    actual_density = density/vector_J

    vector_Jinv = np.reciprocal(vector_J)
    
    actual_kappa = kappa * (vector_J + vector_Jinv)/2

    vector_dt = coeff_cfl * vector_h / np.sqrt((actual_kappa + 4/3*mu)/actual_density)
    delta_t = fdrk.Constant(min(vector_dt))

    return delta_t


def compute_min_mesh_size(mesh):
    DG0_space = fdrk.FunctionSpace(mesh, 'DG', 0)
    diameters = fdrk.CellSize(mesh)

    # v_DG0 = fdrk.TestFunction(mesh) 
    # hvol_form = v_DG0 * diameters * fdrk.dx
    # volume_form = v_DG0 * fdrk.dx

    # vector_volh = fdrk.assemble(hvol_form).vector().get_local()
    # vector_vol = fdrk.assemble(volume_form).vector().get_local()
    # vector_h = vector_volh / vector_vol

    vector_h = fdrk.assemble(fdrk.interpolate(diameters, DG0_space)).vector().get_local()
    
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


def compute_min_max_mesh(mesh: fdrk.MeshGeometry, previous_list_min_max):

    dim = mesh.geometric_dimension()
    assert len(previous_list_min_max)==dim

    list_tuple = []

    for i in range(dim):
        previous_min, previous_max = previous_list_min_max[i]
        coordinates = mesh.coordinates.dat.data[:, i]

        current_min = min(coordinates)
        current_max = max(coordinates)

        if current_min < previous_min:
            actual_min = current_min
        else:
            actual_min = previous_min

        if current_max > previous_max:
            actual_max = current_max
        else:
            actual_max = previous_max

        list_tuple.append([actual_min, actual_max])

    return list_tuple


def sym_grad(vector):
    return 0.5 * (fdrk.nabla_grad(vector) + fdrk.nabla_grad(vector).T)