# CI Optimizer
This repository contains the code for the Cohort Intelligence algorithm, which is used as an optimizer for various mathematical and optimization problems.

The folder structure of the repository is as follows:

 - **/assets** - Contains the graphs of convergence for various test functions and the number of iterations for which the code was run.

 - The individual `.py` files contain the code for the implementation of various test functions in optimization and also the logic for applying Cohort Intelligence to said mathematical functions.
 - The list of the functions implemented:
   1. Sphere
   2. Matyas
   3. Levy
   4. AWJM (referenced from https://doi.org/10.1007/978-981-13-1822-1_43)


Requirements:

 - NumPy --> `pip install numpy`


To run the code and generate the graphs, simply run the python scripts for the required functions.

 - e.g. `python3 sphere.py`

