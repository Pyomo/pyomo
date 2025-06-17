Installation Instructions
=========================

parmest is included in Pyomo (pyomo/contrib/parmest).
To run parmest, you will need Python version 3.x along with 
various Python package dependencies and the IPOPT software 
library for non-linear optimization.

Python package dependencies
---------------------------

#. numpy
#. pandas
#. pyomo
#. mpisppy (optional)
#. matplotlib (optional)
#. scipy.stats (optional)
#. seaborn (optional)
#. mpi4py.MPI (optional)

IPOPT
-----

The IPOPT project homepage is https://github.com/coin-or/Ipopt

Testing
-------

The following commands can be used to test parmest::
  
   cd pyomo/contrib/parmest/tests
   python test_parmest.py

