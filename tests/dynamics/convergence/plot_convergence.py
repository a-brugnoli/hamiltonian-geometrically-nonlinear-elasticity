import numpy as np
import os
import matplotlib.pyplot as plt
import src.postprocessing.options

absolute_path = os.path.dirname(os.path.abspath(__file__))

directory_results = f"{absolute_path}/results/convergence_first_mode/"

n_elem_vec = [4, 8, 16, 30]

time_convergence = []
disp_convergence = []
vel_convergence = []

plt.figure()
plt.title('Displacement at center')
plt.xlabel('Time')

for n_el in n_elem_vec:

    time = np.load(f'{directory_results}time_{n_el}.npy')
    displacement = np.load(f'{directory_results}displacement_{n_el}.npy')

    plt.plot(time, displacement, label=f"$N$ = {n_el}")


plt.legend()

plt.show()