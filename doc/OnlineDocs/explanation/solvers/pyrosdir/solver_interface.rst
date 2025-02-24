.. _pyros_solver_interface:

PyROS Solver Interface
======================

Like other Pyomo solvers, the PyROS solver can be instantiated directly
or through the Pyomo :class:`~pyomo.opt.base.solvers.SolverFactory`:

.. code::

  >>> import pyomo.environ as pyo
  >>> import pyomo.contrib.pyros as pyros  # register the PyROS solver
  >>> pyros_solver = pyo.SolverFactory("pyros")

Subsequently, the solver in invoked by calling the
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method,
the required inputs of which are the:

* Deterministic optimization model
* First-stage ("design") variables
* Second-stage ("control") variables
* Parameters considered uncertain
* Uncertainty set
* Subordinate local and global nonlinear programming (NLP) solvers

See the :py:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
method for further information.

.. note::
    Any variables in the model not specified to be first-stage or second-stage
    variables are automatically considered to be state variables.
