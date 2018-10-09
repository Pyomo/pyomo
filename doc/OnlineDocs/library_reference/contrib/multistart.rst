Multistart Solver
==================

The multistart solver is used in cases where the objective function is known
to be non-convex but the global optimum is still desired. It works by running a non-linear
solver of your choice multiple times at different starting points, and
returns the best of the solutions.


Using Multistart Solver
-----------------------
To use the multistart solver, define your Pyomo model as usual:

.. doctest::

  Required import
  >>> from pyomo.environ import *

  Create a simple model
  >>> m = ConcreteModel()
  >>> m.x = Var()
  >>> m.y = Var()
  >>> m.obj = Objective(expr=m.x**2 + m.y**2)
  >>> m.c = Constraint(expr=m.y >= -2*m.x + 5)

  Invoke the multistart solver
  >>> SolverFactory('multistart').solve(m)  # doctest: +SKIP


Multistart wrapper implementation and optional arguments
--------------------------------------------------------

.. autoclass:: pyomo.contrib.multistart.multi.MultiStart
    :members:
