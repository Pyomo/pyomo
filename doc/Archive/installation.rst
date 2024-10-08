Installation
------------

Pyomo currently supports the following versions of Python:

* CPython: 3.8, 3.9, 3.10, 3.11, 3.12
* PyPy: 3

At the time of the first Pyomo release after the end-of-life of a minor Python
version, Pyomo will remove testing for that Python version.

Using CONDA
~~~~~~~~~~~

We recommend installation with ``conda``, which is included with the
Anaconda distribution of Python. You can install Pyomo in your system
Python installation by executing the following in a shell:

::
   
   conda install -c conda-forge pyomo

Optimization solvers are not installed with Pyomo, but some open source
optimization solvers can be installed with ``conda`` as well:

::

   conda install -c conda-forge ipopt glpk


Using PIP
~~~~~~~~~

The standard utility for installing Python packages is ``pip``.  You
can install Pyomo in your system Python installation by executing
the following in a shell:

::

   pip install pyomo


Conditional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Extensions to Pyomo, and many of the contributions in ``pyomo.contrib``,
often have conditional dependencies on a variety of third-party Python
packages including but not limited to: matplotlib, networkx, numpy,
openpyxl, pandas, pint, pymysql, pyodbc, pyro4, scipy, sympy, and
xlrd. 

A full list of conditional dependencies can be found in Pyomo's
``setup.py`` and displayed using:

::

   python setup.py dependencies --extra optional

Pyomo extensions that require any of these packages will generate
an error message for missing dependencies upon use.

When using *pip*, all conditional dependencies can be installed at once
using the following command:

::

   pip install 'pyomo[optional]'

When using *conda*, many of the conditional dependencies are included
with the standard Anaconda installation.

You can check which Python packages you have installed using the command
``conda list`` or ``pip list``. Additional Python packages may be
installed as needed.


Installation with Cython
~~~~~~~~~~~~~~~~~~~~~~~~

Users can opt to install Pyomo with
`cython <https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html>`_
initialized.

.. note::
   This can only be done via ``pip`` or from source.

Via ``pip``:

::

   pip install pyomo --global-option="--with-cython"

From source (recommended for advanced users only):

::

   git clone https://github.com/Pyomo/pyomo.git
   cd pyomo
   python setup.py install --with-cython
