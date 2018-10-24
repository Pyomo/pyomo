GDPopt logic-based solver
=========================

The GDPopt solver in Pyomo allows users to solve nonlinear Generalized
Disjunctive Programming (GDP) models using logic-based decomposition
approaches, as opposed to the conventional approach via reformulation to a
Mixed Integer Nonlinear Programming (MINLP) model.

GDPopt currently implements an updated version of the logic-based outer
approximation (LOA) algorithm originally described in Turkay & Grossmann, 1996
(`ref <https://dx.doi.org/10.1016/0098-1354(95)00219-7>`_). Usage and
implementation details for GDPopt can be found in the PSE 2018 paper Chen et
al., 2018 (`ref <https://doi.org/10.1016/B978-0-444-64241-7.50143-9>`_,
`preprint <http://egon.cheme.cmu.edu/Papers/Chen_Pyomo_GDP_PSE2018.pdf>`_)

The paper contains the following flowchart, taken from the preprint version:

.. image:: gdpopt_flowchart.png
    :scale: 70%

Usage of GDPopt to solve a Pyomo.GDP concrete model involves:

.. code::

  >>> SolverFactory('gdpopt').solve(model)

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

GDPopt implementation and optional arguments
--------------------------------------------

.. warning:: 

   GDPopt optional arguments should be considered beta code and are
   subject to change.

.. autoclass:: pyomo.contrib.gdpopt.GDPopt.GDPoptSolver
    :members:
