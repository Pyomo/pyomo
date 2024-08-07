.. _gdpopt-main-page:

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

GDPopt can be used to solve a Pyomo.GDP concrete model in two ways.
The simplest is to instantiate the generic GDPopt solver and specify the desired algorithm as an argument to the ``solve`` method:

.. code::

  >>> SolverFactory('gdpopt').solve(model, algorithm='LOA')

The alternative is to instantiate an algorithm-specific GDPopt solver:

.. code::

  >>> SolverFactory('gdpopt.loa').solve(model)

In the above examples, GDPopt uses the GDPopt-LOA algorithm.
Other algorithms may be used by specifying them in the ``algorithm`` argument when using the generic solver or by instantiating the algorithm-specific GDPopt solvers. All GDPopt options are listed below.

.. note::

  The generic GDPopt solver allows minimal configuration outside of the arguments to the ``solve`` method. To avoid repeatedly specifying the same configuration options to the ``solve`` method, use the algorithm-specific solvers.

Logic-based Outer Approximation (LOA)
-------------------------------------

`Chen et al., 2018`_ contains the following flowchart, taken from the preprint version:

.. image:: gdpopt_flowchart.png
    :scale: 70%

An example that includes the modeling approach may be found below.

.. doctest::
  :skipif: not glpk_available

  Required imports
  >>> from pyomo.environ import *
  >>> from pyomo.gdp import *

  Create a simple model
  >>> model = ConcreteModel(name='LOA example')

  >>> model.x = Var(bounds=(-1.2, 2))
  >>> model.y = Var(bounds=(-10,10))
  >>> model.c = Constraint(expr= model.x + model.y == 1)

  >>> model.fix_x = Disjunct()
  >>> model.fix_x.c = Constraint(expr=model.x == 0)

  >>> model.fix_y = Disjunct()
  >>> model.fix_y.c = Constraint(expr=model.y == 0)

  >>> model.d = Disjunction(expr=[model.fix_x, model.fix_y])
  >>> model.objective = Objective(expr=model.x + 0.1*model.y, sense=minimize)

  Solve the model using GDPopt
  >>> results = SolverFactory('gdpopt.loa').solve(
  ...     model, mip_solver='glpk') # doctest: +IGNORE_RESULT

  Display the final solution
  >>> model.display()
  Model LOA example
  <BLANKLINE>
    Variables:
      x : Size=1, Index=None
          Key  : Lower : Value : Upper : Fixed : Stale : Domain
          None :  -1.2 :     0 :     2 : False : False :  Reals
      y : Size=1, Index=None
          Key  : Lower : Value : Upper : Fixed : Stale : Domain
          None :   -10 :     1 :    10 : False : False :  Reals
  <BLANKLINE>
    Objectives:
      objective : Size=1, Index=None, Active=True
          Key  : Active : Value
          None :   True :   0.1
  <BLANKLINE>
    Constraints:
      c : Size=1
          Key  : Lower : Body : Upper
          None :   1.0 :    1 :   1.0

.. note:: 

   When troubleshooting, it can often be helpful to turn on verbose
   output using the ``tee`` flag.

.. code::

  >>> SolverFactory('gdpopt.loa').solve(model, tee=True)

Global Logic-based Outer Approximation (GLOA)
---------------------------------------------

The same algorithm can be used to solve GDPs involving nonconvex nonlinear constraints by solving the subproblems globally:

.. code::

  >>> SolverFactory('gdpopt.gloa').solve(model)

.. warning::

  The ``nlp_solver`` option must be set to a global solver for the solution returned by GDPopt to also be globally optimal.

Relaxation with Integer Cuts (RIC)
----------------------------------

Instead of outer approximation, GDPs can be solved using the same MILP relaxation as in the previous two algorithms, but instead of using the subproblems to generate outer-approximation cuts, the algorithm adds only no-good cuts for every discrete solution encountered:

.. code::

  >>> SolverFactory('gdpopt.ric').solve(model)

Again, this is a global algorithm if the subproblems are solved globally, and is not otherwise.

.. note::

  The RIC algorithm will not necessarily enumerate all discrete solutions as it is possible for the bounds to converge first. However, full enumeration is not uncommon.

Logic-based Branch-and-Bound (LBB)
----------------------------------

The GDPopt-LBB solver branches through relaxed subproblems with inactive disjunctions.
It explores the possibilities based on best lower bound,
eventually activating all disjunctions and presenting the globally optimal solution.

To use the GDPopt-LBB solver, define your Pyomo GDP model as usual:

.. doctest::
  :skipif: not baron_available

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

  >>> results = SolverFactory('gdpopt.lbb').solve(m)
  WARNING: 09/06/22: The GDPopt LBB algorithm currently has known issues. Please
      use the results with caution and report any bugs!

  >>> print(results)  # doctest: +SKIP
  >>> print(results.solver.status)
  ok
  >>> print(results.solver.termination_condition)
  optimal

  >>> print([value(m.y1.indicator_var), value(m.y2.indicator_var)])
  [True, False]

GDPopt implementation and optional arguments
--------------------------------------------

.. warning:: 

   GDPopt optional arguments should be considered beta code and are
   subject to change.

.. autoclass:: pyomo.contrib.gdpopt.GDPopt.GDPoptSolver
    :members:

.. autoclass:: pyomo.contrib.gdpopt.loa.GDP_LOA_Solver
    :members:

.. autoclass:: pyomo.contrib.gdpopt.gloa.GDP_GLOA_Solver
    :members:

.. autoclass:: pyomo.contrib.gdpopt.ric.GDP_RIC_Solver
    :members:

.. autoclass:: pyomo.contrib.gdpopt.branch_and_bound.GDP_LBB_Solver
    :members:
