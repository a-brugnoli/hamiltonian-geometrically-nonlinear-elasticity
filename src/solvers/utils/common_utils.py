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
