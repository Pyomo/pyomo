Using Pyomo on the p-median problem.
====================================

The goal of this problem is to place facilities to meet customer demands.
This formulation randomly generates customer demands.


Solving with AMPL
-----------------
Files
  pmedian.mod     - The AMPL model
  pmedian.dat     - A simple data file for this example

Running AMPL
  1. Add the directory containing 'cplexamp' to your PATH environment
  2. ampl pmedian.ampl


Solving with Pyomo
------------------
Files
  pmedian.py      - A Python model written with Pyomo objects
  pmedian.dat     - A simple data file for this example

Running Pyomo
  pyomo pmedian.py pmedian.dat


Using Pyomo with a customized solver
====================================

Files
  solver1.py      - A Python file that contains a greedy heuristic
  solver2.py      - A Python file that contains a randomized heuristic
  pmedian.py      - A Python model written with Pyomo objects
  pmedian.dat     - A simple data file for this example

Running Pyomo
  pyomo --preprocess=solver1.py --solver=greedy pmedian.py pmedian.dat
  pyomo --preprocess=solver2.py --solver=random pmedian.py pmedian.dat

