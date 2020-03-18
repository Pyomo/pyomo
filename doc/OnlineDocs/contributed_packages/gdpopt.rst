GDPopt logic-based solver
=========================

The GDPopt solver in Pyomo allows users to solve nonlinear Generalized
Disjunctive Programming (GDP) models using logic-based decomposition
approaches, as opposed to the conventional approach via reformulation to a
Mixed Integer Nonlinear Programming (MINLP) model.

The main advantage of these techniques is their ability to solve subproblems
in a reduced space, including nonlinear constraints only for ``True`` logical blocks.
As a result, GDPopt is most effective for nonlinear GDP models.

Three algorithms are available in GDPopt:

1. Logic-based outer approximation (LOA) [`Turkay & Grossmann, 1996`_]
2. Global logic-based outer approximation (GLOA) [`Lee & Grossmann, 2001`_]
3. Logic-based branch-and-bound (LBB) [`Lee & Grossmann, 2001`_]

Usage and implementation details for GDPopt can be found in the PSE 2018 paper
(`Chen et al., 2018`_), or via its
`preprint <http://egon.cheme.cmu.edu/Papers/Chen_Pyomo_GDP_PSE2018.pdf>`_.

Credit for prototyping and development can be found in the ``GDPopt`` class documentation, below.

.. _Turkay & Grossmann, 1996: https://dx.doi.org/10.1016/0098-1354(95)00219-7
.. _Lee & Grossmann, 2001: https://doi.org/10.1016/S0098-1354(01)00732-3
.. _Lee & Grossmann, 2000: https://doi.org/10.1016/S0098-1354(00)00581-0
.. _Chen et al., 2018: https://doi.org/10.1016/B978-0-444-64241-7.50143-9

Usage of GDPopt to solve a Pyomo.GDP concrete model involves:

.. code::

  >>> SolverFactory('gdpopt').solve(model)

.. note::

  By default, GDPopt uses the GDPopt-LOA strategy.
  Other strategies may be used by specifying the ``strategy`` argument during ``solve()``.
  All GDPopt options are listed below.

Logic-based Outer Approximation
-------------------------------

`Chen et al., 2018`_ contains the following flowchart, taken from the preprint version:

.. image:: gdpopt_flowchart.png
    :scale: 70%

An example which includes the modeling approach may be found below.

.. doctest::

  Required imports
  >>> from pyomo.environ import *
  >>> from pyomo.gdp import *

  Create a simple model
  >>> model = ConcreteModel()

  >>> model.x = Var(bounds=(-1.2, 2))
  >>> model.y = Var(bounds=(-10,10))

  >>> model.fix_x = Disjunct()
  >>> model.fix_x.c = Constraint(expr=model.x == 0)

  >>> model.fix_y = Disjunct()
  >>> model.fix_y.c = Constraint(expr=model.y == 0)

  >>> model.c = Disjunction(expr=[model.fix_x, model.fix_y])
  >>> model.objective = Objective(expr=model.x, sense=minimize)

  Solve the model using GDPopt
  >>> SolverFactory('gdpopt').solve(model, mip_solver='glpk') # doctest: +SKIP

The solution may then be displayed by using the commands

.. code::

  >>> model.objective.display()
  >>> model.display()
  >>> model.pprint()

.. note:: 

   When troubleshooting, it can often be helpful to turn on verbose
   output using the ``tee`` flag.

.. code::

  >>> SolverFactory('gdpopt').solve(model, tee=True)

Logic-based Branch-and-Bound
----------------------------

The GDPopt-LBB solver branches through relaxed subproblems with inactive disjunctions.
It explores the possibilities based on best lower bound,
eventually activating all disjunctions and presenting the globally optimal solution.

To use the GDPopt-LBB solver, define your Pyomo GDP model as usual:

.. doctest::

  Required imports
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

  Invoke the GDPopt-LBB solver
  >>> results = SolverFactory('gdpopt').solve(m, strategy='LBB')

  >>> print(results)  # doctest: +SKIP
  >>> print(results.solver.status)
  ok
  >>> print(results.solver.termination_condition)
  optimal

  >>> print([value(m.y1.indicator_var), value(m.y2.indicator_var)])
  [1, 0]

GDPopt implementation and optional arguments
--------------------------------------------

.. warning:: 

   GDPopt optional arguments should be considered beta code and are
   subject to change.

.. autoclass:: pyomo.contrib.gdpopt.GDPopt.GDPoptSolver
    :members:
