import time
import pickle
import numpy as np
from experiments.vonkarman_beam.parameters_vonkarman import *

beam.set_time_step(dt_base/2**3)
dict_results_linear = beam.leapfrog(save_vars=True, linear=True)

with open(file_results_linear, "wb") as f:
        pickle.dump(dict_results_linear, f)