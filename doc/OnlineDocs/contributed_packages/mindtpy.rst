MindtPy solver
==============

The Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy) solver
allows users to solve Mixed-Integer Nonlinear Programs (MINLP) using decomposition algorithms.
These decomposition algorithms usually rely on the solution of Mixed-Intger Linear Programs
(MILP) and Nonlinear Programs (NLP).

MindtPy currently implements the Outer Approximation (OA) algorithm originally described in
`Duran & Grossmann, 1986`_. Usage and implementation
details for MindtPy can be found in the PSE 2018 paper Bernal et al.,
(`ref <https://doi.org/10.1016/B978-0-444-64241-7.50144-0>`_,
`preprint <http://egon.cheme.cmu.edu/Papers/Bernal_Chen_MindtPy_PSE2018Paper.pdf>`_).

.. _Duran & Grossmann, 1986: https://dx.doi.org/10.1007/BF02592064

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

  >>> model.c1 = Constraint(expr=(model.x-4.0)**2 - model.x <= 50.0*(1-model.y))
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

Single tree implementation
---------------------------------------------

MindtPy also supports single tree implementation of Outer Approximation (OA) algorithm, which is known as LP/NLP algorithm originally described in `Quesada & Grossmann`_.
The LP/NLP algorithm in MindtPy is implemeted based on the LazyCallback function in commercial solvers.

.. _Quesada & Grossmann: https://www.sciencedirect.com/science/article/abs/pii/0098135492800288


.. Note::

The single tree implementation currently only works with CPLEX.
To use LazyCallback function of CPLEX from Pyomo, the `CPLEX Python API`_ is required.
This means both IBM ILOG CPLEX Optimization Studio and the CPLEX-Python modules should be installed on your computer.


.. _CPLEX Python API: https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html


A usage example for single tree is as follows:

.. code::

  >>> import pyomo.environ as pyo
  >>> model = pyo.ConcreteModel()

  >>> model.x = pyo.Var(bounds=(1.0, 10.0), initialize=5.0)
  >>> model.y = pyo.Var(within=Binary)

  >>> model.c1 = Constraint(expr=(model.x-4.0)**2 - model.x <= 50.0*(1-model.y))
  >>> model.c2 = pyo.Constraint(expr=model.x*log(model.x)+5.0 <= 50.0*(model.y))
  
  >>> model.objective = pyo.Objective(expr=model.x, sense=pyo.minimize)

  Solve the model using single tree implementation in MindtPy
  >>> pyo.SolverFactory('mindtpy').solve(
  ...    model, strategy='OA',
  ...    mip_solver='cplex_persistent', nlp_solver='ipopt', single_tree=True)
  >>> model.objective.display()




MindtPy implementation and optional arguments
---------------------------------------------

.. warning::

   MindtPy optional arguments should be considered beta code and are
   subject to change.

.. autoclass:: pyomo.contrib.mindtpy.MindtPy.MindtPySolver
    :members:
