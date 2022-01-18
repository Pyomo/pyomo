Installation
------------

Pyomo currently supports the following versions of Python:

* CPython: 3.6, 3.7, 3.8, 3.9
* PyPy: 3


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
packages including but not limited to: matplotlib, networkx, numpy,
openpyxl, pandas, pint, pymysql, pyodbc, pyro4, scipy, sympy, and
xlrd. (A full list of conditional dependencies can be found in Pyomo's
`setup.py`)

Pyomo extensions that require any of these packages will generate
an error message for missing dependencies upon use.

Many of the conditional dependencies are already distributed with
Anaconda. You can check which Python packages you already have installed
using the command ``conda list`` or ``pip list``. Additional Python
packages may be installed as needed.

Alternatively, you can install all conditional dependencies using pip:

::

   pip install 'pyomo[optional]'


