.. _pyros_solver_interface:

PyROS Solver Interface
======================

The PyROS solver is invoked through the
:py:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
method of the PyROS solver class.
In summary, the required inputs to the PyROS solver are:

* The deterministic optimization model
* List of first-stage ("design") variables
* List of second-stage ("control") variables
* List of parameters considered uncertain
* The uncertainty set
* Subordinate local and global nonlinear programming (NLP) solvers

.. note::
    Any variables in the model not specified to be first-stage or second-stage
    variables are automatically considered to be state variables.


The PyROS solver can be instantiated through the Pyomo
:class:`~pyomo.opt.base.solvers.SolverFactory`:

.. code::

  >>> import pyomo.environ as pyo
  >>> import pyomo.contrib.pyros as pyros  # register the PyROS solver
  >>> pyros_solver = pyo.SolverFactory("pyros")
