Installation Instructions
=========================

parmest is included in Pyomo (pyomo/contrib/parmest).
To run parmest, you will need Python version 3.x along with 
various Python package dependencies and the IPOPT software 
library for non-linear optimization.

Python Package Dependencies
---------------------------

1.  Install NumPy and Pandas with your preferred package manager;
    both NumPy and SciPy are required dependencies of parmest.
    You may install NumPy and Pandas with, for example, ``conda``:

    ::

        conda install numpy pandas

    or ``pip``:

    ::

        pip install numpy pandas
2.  `Install Pyomo <https://pyomo.readthedocs.io/en/stable/getting_started/installation.html#pyomo-installation>`_.
    parmest is included in the Pyomo software package, at pyomo/contrib/parmest.
3.  (*Optional*) Install ``matplotlib`` and ``scipy``:

    ::

        pip install scipy matplotlib
4.  (*Optional*) Install ``seaborn``

    ::

        pip install seaborn

IPOPT
-----

The IPOPT project homepage is https://github.com/coin-or/Ipopt

Testing
-------

The following commands can be used to test parmest::
  
   cd pyomo/contrib/parmest/tests
   python test_parmest.py

