from parameters import *
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from src.postprocessing.options import configure_matplotib
configure_matplotib()


dt = time_step_vec[0]
bending_column.set_time_step(dt)

t0_lin_implicit_leapfrog = time.perf_counter()
dict_results_leapfrog = bending_column.leapfrog(save_vars=True)
tf_lin_implicit_leapfrog = time.perf_counter()

disp_array_leapfrog = dict_results_leapfrog["displacement"]
vel_array_leapfrog = dict_results_leapfrog["velocity"]
array_coordinates = bending_column.domain.coordinates.dat.data

distances = np.sum((array_coordinates - x_point)**2, axis=1)
    
# Find index of minimum distance
closest_idx = np.argmin(distances)

print(closest_idx)

disp_at_point = disp_array_leapfrog[:, closest_idx, :]

disp_at_point_x = disp_at_point[:, 0]
disp_at_point_y = disp_at_point[:, 1]
disp_at_point_z = disp_at_point[:, 2]

plt.figure()
plt.plot(t_vec_output_ms, disp_at_point_x, label="x")
plt.plot(t_vec_output_ms, disp_at_point_y, label="y")
plt.plot(t_vec_output_ms, disp_at_point_z, label="z")
plt.xlabel("Time [ms]")
plt.title("Displacement")
plt.legend()
plt.show()