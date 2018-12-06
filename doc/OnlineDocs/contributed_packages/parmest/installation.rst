Installation Instructions
=========================

The parmest software is included in Pyomo.
To run parmest, you will need Python version 3.x along with 
various Python package dependencies and the IPOPT software 
library for non-linear optimization.

Python package dependencies
-------------------------------

#. numpy
#. pandas
#. pyomo
#. pyutilib
#. matplotlib (optional, used for graphics)
#. scipy.stats (optional, used for graphics)
#. seaborn (optional, used for graphics)
#. mpi4py.MPI (optional, used for parallel computing)

IPOPT
-------
IPOPT can be downloaded from https://projects.coin-or.org/Ipopt. 

Testing
-------

Use these (Unix) terminal commands to test parmest::
  
   cd pyomo/contrib/parmest/tests
   python test_parmest.py

