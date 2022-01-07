.. _api_documentation:

APPSI
=====

Auto-Persistent Pyomo Solver Interfaces

.. automodule:: pyomo.contrib.appsi
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

    >>> import pyomo.environ as pe
    >>> from pyomo.contrib import appsi
    >>> import numpy as np
    >>> from pyomo.common.timing import HierarchicalTimer
    >>> m = pe.ConcreteModel()
    >>> m.x = pe.Var()
    >>> m.y = pe.Var()
    >>> m.p = pe.Param(mutable=True)
    >>> m.obj = pe.Objective(expr=m.x**2 + m.y**2)
    >>> m.c1 = pe.Constraint(expr=m.y >= pe.exp(m.x))
    >>> m.c2 = pe.Constraint(expr=m.y >= (m.x - m.p)**2)
    >>> opt = appsi.solvers.Ipopt()
    >>> timer = HierarchicalTimer()
    >>> for p_val in np.linspace(1, 10, 100):
    >>>     m.p.value = float(p_val)
    >>>     res = opt.solve(m, timer=timer)
    >>>     assert res.termination_condition == appsi.base.TerminationCondition.optimal
    >>>     print(res.best_feasible_objective)
    >>> print(timer)

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

.. code-block::

   cd pyomo/contrib/appsi/
   python build.py
