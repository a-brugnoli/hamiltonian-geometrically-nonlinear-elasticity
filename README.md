Code to implement a linearly-implicit time integrator using a non-canonical Hamiltonian formulation.

The numerical examples can be reproduced as follows:
* install the Zenodo-archived release of firedrake hosted at [https://zenodo.org/records/15009425](https://zenodo.org/records/15009425)
  ```
  curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-configure
  firedrake-install --doi 10.5281/zenodo.15009425
  ```
* actiivate the environment and run ```pip install tqdm time pickle``` to install required librairies.
* To plot the results for the Duffing oscillator run
  ```
  $ python experiments/duffing/plot_convergence_duffing.py
  ```

* To plot the results for the von-Kàrmàn beam run
  ```
  $ python experiments/vonkarman_beam/plot_convergence_vonkarman.py
  ```
  for the convergence test and
  ```
  $ python experiments/vonkarman_beam/plot_signals_vonkarman.py 
  ```
  for the plot of the reference solutions and some time series (energy and displacement at a point).

* The array containing the results for finite strain elasticity are not included as they are quite heavy.
  One can visualize the images in the folder `experiments/finite_strain_elasticity/images/`
  To repricate the results run 
  ```
  $ python experiments/finite_strain_elasticity/collect_results_elasticity.py
  ```
  This will create two folders in your home directory named 
  - `StoreResults/FiniteStrainElasticity/leapfrog`
  - `StoreResults/FiniteStrainElasticity/results`
  
  The first contains the paraview files for the reference solution to the problem and the second contains the results from the convergence test.

