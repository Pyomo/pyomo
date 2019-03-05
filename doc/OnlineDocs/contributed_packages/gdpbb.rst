GDP Branch and Bound Solver
===========================

The GDP Branch and Bound solver is used to solve Generalized Disjunctive
Programming (GDP) Problems. It branches through relaxed subproblems with
inactive disjunctions. It explores the possibilities based on best lower bound,
eventually activating all disjunctions and presenting the global optimal.


Using GDP Branch and Bound Solver
---------------------------------
To use the GDPbb solver, define your Pyomo GDP model as usual:

.. doctest::

  Required import
  >>> from pyomo.environ import *
  >>> from pyomo.gdp import Disjunct, Disjunction

  Create a simple model
  >>> m = ConcreteModel()
  >>> m.x1 = Var(bounds = (0,8))
  >>> m.x2 = Var(bounds = (0,8))
  >>> m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
  >>> m.y1 = Disjunct()
  >>> m.y2 = Disjunct()
  >>> m.y1.c1 = Constraint(expr=m.x1 >= 2)
  >>> m.y1.c2 = Constraint(expr=m.x2 >= 2)
  >>> m.y2.c1 = Constraint(expr=m.x1 >= 3)
  >>> m.y2.c2 = Constraint(expr=m.x2 >= 3)
  >>> m.djn = Disjunction(expr=[m.y1, m.y2])

  Invoke the GDPbb solver
  >>> results = SolverFactory('gdpbb').solve(m)

  >>> print(results)  # doctest: +SKIP
  >>> print(results.solver.status)
  ok
  >>> print(results.solver.termination_condition)
  optimal

  >>> print([value(m.y1.indicator_var), value(m.y2.indicator_var)])
  [1, 0]


GDP Branch and Bound implementation and optional arguments
----------------------------------------------------------

.. autoclass:: pyomo.contrib.gdpbb.GDPbb.GDPbbSolver
    :members:
