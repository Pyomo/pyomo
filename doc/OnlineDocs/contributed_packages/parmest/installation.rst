Installation Instructions
=========================

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

For each package, you need to cd to the directory, then give the
terminal command::
  
    python setup.py install
	
IPOPT
-------
IPOPT can be downloaded from https://projects.coin-or.org/Ipopt. 

Testing
-------

Use these terminal commands to test parmest::
  
   cd parmest/parmest/tests
   python test_parmest.py

