Installation
------------

Pyomo currently supports the following versions of Python:

* CPython: 2.7, 3.4, 3.5, 3.6, 3.7, 3.8
* PyPy: 2, 3


Using CONDA
~~~~~~~~~~~

We recommend installation with *conda*, which is included with the
Anaconda distribution of Python. You can install Pyomo in your system
Python installation by executing the following in a shell:

::
   
   conda install -c conda-forge pyomo

Optimization solvers are not installed with Pyomo, but some open source
optimization solvers can be installed with conda as well:

::

   conda install -c conda-forge ipopt glpk


Using PIP
~~~~~~~~~

The standard utility for installing Python packages is *pip*.  You
can install Pyomo in your system Python installation by executing
the following in a shell:

::

   pip install pyomo


Conditional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Extensions to Pyomo, and many of the contributions in `pyomo.contrib`,
also have conditional dependencies on a variety of third-party Python
packages including but not limited to: numpy, scipy, sympy, networkx,
openpxl, pyodbc, xlrd, pandas, matplotlib, pymysql, pyro4, and
pint. Pyomo extensions that require any of these packages will generate
an error message for missing dependencies upon use.

Many of the conditional dependencies are already distributed with
Anaconda. You can check which Python packages you already have installed
using the command ``conda list`` or ``pip list``. Additional Python
packages may be installed as needed.
