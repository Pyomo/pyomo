Installation
------------

Pyomo currently supports the following versions of Python:

* CPython: 2.7, 3.4, 3.5, 3.6


Using CONDA
~~~~~~~~~~~

We recommend installation with *conda*, which is included with the
Anaconda distribution of Python. You can install Pyomo in your system
Python installation by executing the following in a shell:

::
   
   conda install -c conda-forge pyomo

Pyomo also has conditional dependencies on a variety of third-party
Python packages.  These can also be installed with conda:

::

   conda install -c conda-forge pyomo.extras

Optimization solvers are not installed with Pyomo, but some open source
optimization solvers can be installed with conda as well:

::

   conda install -c conda-forge ipopt coincbc glpk


Using PIP
~~~~~~~~~

The standard utility for installing Python packages is *pip*.  You
can install Pyomo in your system Python installation by executing
the following in a shell:

::

   pip install pyomo
