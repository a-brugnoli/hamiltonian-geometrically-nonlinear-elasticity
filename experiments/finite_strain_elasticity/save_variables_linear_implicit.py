import time
import pickle
import numpy as np
from experiments.finite_strain_elasticity.parameters_elasticity import *

bending_column.set_time_step(dt_base)
dict_results_reference = bending_column.linear_implicit_static_condensation(save_vars=False,\
                                                        paraview_directory=paraview_directory)
