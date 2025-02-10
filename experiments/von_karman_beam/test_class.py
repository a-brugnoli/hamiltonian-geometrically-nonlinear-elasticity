import firedrake as fdrk
from von_karman_beam import VonKarmanBeam

beam = VonKarmanBeam()  

mesh = beam.domain
x, = fdrk.SpatialCoordinate(mesh)
q_x_0 = fdrk.Constant(0)
q_z_0 = x**3


eps_0 = beam.bending_stress(q_z_0)

