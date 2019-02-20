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
  >>> m.x = Var([1,2], bounds=(0,8))
  >>> m.obj = Objective(expr=sum(m.x[:]), sense=minimize)
  >>> m.y = Disjunct([1,2])
  >>> m.y[1].c = Constraint([1,2], rule=lambda d,i: m.x[i] >= 2)
  >>> m.y[2].c = Constraint([1,2], rule=lambda d,i: m.x[i] >= 3)
  >>> m.djn = Disjunction(expr=m.y[:])

  Invoke the GDPbb solver
  >>> results = SolverFactory('gdpbb').solve(m)

  >>> print(results)  # doctest: +SKIP
  >>> print(results.solver.status)
  ok
  >>> print(results.solver.termination_condition)
  optimal

  >>> print([value(m.y[i].indicator_var) for i in (1, 2)])
  [0, 1]


GDP Branch and Bound implementation and optional arguments
----------------------------------------------------------

.. autoclass:: pyomo.contrib.gdpbb.gdpbb.GDPbbSolver
    :members:
