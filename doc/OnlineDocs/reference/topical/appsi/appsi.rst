.. _api_documentation:

APPSI
=====

Auto-Persistent Pyomo Solver Interfaces

.. automodule:: pyomo.contrib.appsi
    :noindex:
    :members:
    :show-inheritance:

.. toctree::
   
   appsi.base
   appsi.solvers

APPSI solver interfaces are designed to work very similarly to most
Pyomo solver interfaces but are very efficient for resolving the same
model with small changes. This is very beneficial for applications
such as Benders' Decomposition, Optimization-Based Bounds Tightening,
Progressive Hedging, Outer-Approximation, and many others. Here is an
example of using an APPSI solver interface.

.. code-block:: python

    >>> import pyomo.environ as pyo
    >>> from pyomo.contrib import appsi
    >>> import numpy as np
    >>> from pyomo.common.timing import HierarchicalTimer
    >>> m = pyo.ConcreteModel()
    >>> m.x = pyo.Var()
    >>> m.y = pyo.Var()
    >>> m.p = pyo.Param(mutable=True)
    >>> m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
    >>> m.c1 = pyo.Constraint(expr=m.y >= pyo.exp(m.x))
    >>> m.c2 = pyo.Constraint(expr=m.y >= (m.x - m.p)**2)
    >>> opt = appsi.solvers.Ipopt()
    >>> timer = HierarchicalTimer()
    >>> for p_val in np.linspace(1, 10, 100):
    >>>     m.p.value = float(p_val)
    >>>     res = opt.solve(m, timer=timer)
    >>>     assert res.termination_condition == appsi.base.TerminationCondition.optimal
    >>>     print(res.best_feasible_objective)
    >>> print(timer)

Alternatively, you can access the APPSI solvers through the classic
``SolverFactory`` using the pattern ``appsi_solvername``.

.. code-block:: python

    >>> import pyomo.environ as pyo
    >>> opt_ipopt = pyo.SolverFactory('appsi_ipopt')
    >>> opt_highs = pyo.SolverFactory('appsi_highs')

Extra performance improvements can be made if you know exactly what
changes will be made in your model. In the example above, only
parameter values are changed, so we can setup the
:py:class:`~pyomo.contrib.appsi.base.UpdateConfig` so that the solver
does not check for changes in variables or constraints.

.. code-block:: python

    >>> timer = HierarchicalTimer()
    >>> opt.update_config.check_for_new_or_removed_constraints = False
    >>> opt.update_config.check_for_new_or_removed_vars = False
    >>> opt.update_config.update_constraints = False
    >>> opt.update_config.update_vars = False
    >>> for p_val in np.linspace(1, 10, 100):
    >>>     m.p.value = float(p_val)
    >>>     res = opt.solve(m, timer=timer)
    >>>     assert res.termination_condition == appsi.base.TerminationCondition.optimal
    >>>     print(res.best_feasible_objective)
    >>> print(timer)

Solver independent options can be specified with the
:py:class:`~pyomo.contrib.appsi.base.SolverConfig` or derived
classes. For example:

.. code-block:: python

    >>> opt.config.stream_solver = True

Solver specific options can be specified with the
:py:meth:`~pyomo.contrib.appsi.base.Solver.solver_options`
attribute. For example:

.. code-block:: python

    >>> opt.solver_options['max_iter'] = 20

Installation
------------
There are a few ways to install Appsi listed below.

Option1:

.. code-block::

   pyomo build-extensions

Option2:

.. code-block::

   cd pyomo/contrib/appsi/
   python build.py

Option3:

.. code-block::

   python
   >>> from pyomo.contrib.appsi.build import build_appsi
   >>> build_appsi()

