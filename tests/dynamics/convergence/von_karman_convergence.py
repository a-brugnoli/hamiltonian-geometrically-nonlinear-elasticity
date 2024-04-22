import firedrake as fdrk
import numpy as np
from math import ceil
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.solvers.hamiltonian_von_karman import HamiltonianVonKarmanSolver
from src.problems.free_vibrations_von_karman import FirstModeVonKarman
import os, sys

def convergence_von_karman(n_elem):

    pol_degree = 1
    time_step = 5 * 10**(-6)
    T_end = 3 * 10**(-2)
    n_time  = ceil(T_end/time_step)

    amplitude = 10
    problem = FirstModeVonKarman(n_elem, n_elem, amplitude=amplitude)

    solver = HamiltonianVonKarmanSolver(problem, 
                                time_step, 
                                pol_degree, \
                                coupling=True, 
                                membrane_inertia=False)

                
    time_vector = np.linspace(0, T_end, num=n_time+1)
    energy_vector = np.zeros((n_time+1, ))
    energy_vector[0] = fdrk.assemble(solver.energy_old)

    L_x, L_y = problem.parameters["L_x"], problem.parameters["L_y"]
    center = (L_x/2, L_y/2)

    displacement_at_center = np.zeros((n_time+1, ))
    displacement_at_center[0] = solver.bend_displacement_old.at(center)

    velocity_at_center = np.zeros((n_time+1, ))
    velocity_at_center[0] = solver.bend_velocity_old.at(center)


    for ii in tqdm(range(1, n_time+1)):

        solver.integrate()
        energy_vector[ii] = fdrk.assemble(solver.energy_new)

        solver.update_variables()
        displacement_at_center[ii] = solver.bend_displacement_old.at(center)
        velocity_at_center[ii] = solver.bend_velocity_old.at(center)


    return time_vector, displacement_at_center, velocity_at_center

if __name__=="__main__":

    absolute_path = os.path.dirname(os.path.abspath(__file__))
    directory_results = f"{absolute_path}/results/convergence_first_mode/"
    if not os.path.exists(directory_results):
        os.makedirs(directory_results)

    n_elem_vec = [10, 20, 30, 40, 50]

    for n_el in n_elem_vec:

        t_n, disp_n, vel_n = convergence_von_karman(n_el)

        np.save(f'{directory_results}time_{n_el}.npy', t_n)
        np.save(f'{directory_results}displacement_{n_el}.npy', disp_n)
        np.save(f'{directory_results}velocity_{n_el}.npy', vel_n)




