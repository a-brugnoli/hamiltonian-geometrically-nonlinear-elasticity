import firedrake as fdrk
import matplotlib.pyplot as plt

square_hole = fdrk.Mesh("square_hole.msh")

fdrk.triplot(square_hole)

plt.show()