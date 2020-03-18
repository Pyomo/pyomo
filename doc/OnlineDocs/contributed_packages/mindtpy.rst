MindtPy solver
==============

The Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy) solver
allows users to solve Mixed-Integer Nonlinear Programs (MINLP) using decomposition algorithms.
These decomposition algorithms usually rely on the solution of Mixed-Intger Linear Programs
(MILP) and Nonlinear Programs (NLP).

MindtPy currently implements the Outer Approximation (OA) algorithm originally described in
`Duran & Grossmann`_. Usage and implementation
details for MindtPy can be found in the PSE 2018 paper Bernal et al.,
(`ref <https://doi.org/10.1016/B978-0-444-64241-7.50144-0>`_,
`preprint <http://egon.cheme.cmu.edu/Papers/Bernal_Chen_MindtPy_PSE2018Paper.pdf>`_).

.. _Duran & Grossmann: https://dx.doi.org/10.1007/BF02592064

Usage of MindtPy to solve a Pyomo concrete model involves:

.. code::

  >>> SolverFactory('mindtpy').solve(model)

An example which includes the modeling approach may be found below.

.. doctest::

  Required imports
  >>> from pyomo.environ import *

  Create a simple model
  >>> model = ConcreteModel()

  >>> model.x = Var(bounds=(1.0,10.0),initialize=5.0)
  >>> model.y = Var(within=Binary)

  >>> model.c1 = Constraint(expr=(model.x-3.0)**2 <= 50.0*(1-model.y))
  >>> model.c2 = Constraint(expr=model.x*log(model.x)+5.0 <= 50.0*(model.y))

  >>> model.objective = Objective(expr=model.x, sense=minimize)

  Solve the model using MindtPy
  >>> SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt') # doctest: +SKIP

The solution may then be displayed by using the commands

.. code::

  >>> model.objective.display()
  >>> model.display()
  >>> model.pprint()

.. note::

   When troubleshooting, it can often be helpful to turn on verbose
   output using the ``tee`` flag.

.. code::

  >>> SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)

MindtPy implementation and optional arguments
---------------------------------------------

.. warning::

   MindtPy optional arguments should be considered beta code and are
   subject to change.

.. autoclass:: pyomo.contrib.mindtpy.MindtPy.MindtPySolver
    :members:
