MindtPy solver
==============

The Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy) solver
allows users to solve Mixed-Integer Nonlinear Programs (MINLP) using decomposition algorithms.
These decomposition algorithms usually rely on the solution of Mixed-Intger Linear Programs
(MILP) and Nonlinear Programs (NLP).

MindtPy currently implements the Outer Approximation (OA) algorithm originally described in
[`Duran & Grossmann, 1986`_] and the Extended Cutting Plane (ECP) algorithm originally described in [`Westerlund & Petterson, 1995`_]. Usage and implementation
details for MindtPy can be found in the PSE 2018 paper Bernal et al.,
(`ref <https://doi.org/10.1016/B978-0-444-64241-7.50144-0>`_,
`preprint <http://egon.cheme.cmu.edu/Papers/Bernal_Chen_MindtPy_PSE2018Paper.pdf>`_).

.. _Duran & Grossmann, 1986: https://dx.doi.org/10.1007/BF02592064
.. _Westerlund & Petterson, 1995: http://dx.doi.org/10.1016/0098-1354(95)87027-X

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

MindtPy also supports setting options for mip solver and nlp solver. 

.. code::

  >>> SolverFactory('mindtpy').solve(model, 
                                     strategy='OA',
                                     time_limit=3600,
                                     mip_solver='gams',
                                     mip_solver_args=dict(solver='cplex', warmstart=True),
                                     nlp_solver='ipopt',
                                     tee=True)

There are three initialization strategies in MindtPy: rNLP, initial_binary, max_binary. In OA and GOA strategies, the default initialization strategy is rNLP. In ECP strategy, the default initialization strategy is max_binary.

Single tree implementation
---------------------------------------------

MindtPy also supports single tree implementation of Outer Approximation (OA) algorithm, which is known as LP/NLP algorithm originally described in [`Quesada & Grossmann`_].
The LP/NLP algorithm in MindtPy is implemeted based on the LazyCallback function in commercial solvers.

.. _Quesada & Grossmann: https://www.sciencedirect.com/science/article/abs/pii/0098135492800288


.. note::

   The single tree implementation currently only works with CPLEX.  To
   use LazyCallback function of CPLEX from Pyomo, the `CPLEX Python
   API`_ is required.  This means both IBM ILOG CPLEX Optimization
   Studio and the CPLEX-Python modules should be installed on your
   computer.


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


Global Outer Approximation
---------------------------------------------

Apart of the decomposition methods for convex MINLP problems [`Kronqvist et al., 2019`_], MindtPy provides an implementation of Global Outer Approximation (GOA) as described in [`Kesavan et al., 2004`_], to provide optimality guaranteed for nonconvex MINLP problems. Here, the validity of the Mixed-integer Linear Programming relaxation of the original problem is guaranteed via the usage of Generalized McCormick envelopes, computed using the package `MC++`_. The NLP subproblems in this case need to be solved to global optimality, which can be achieved through global NLP solvers such as BARON or SCIP. No-good cuts are added to each iteration, guaranteeing the finite convergence of the algorithm. Notice that this methods are more computationally expensive than the other strategies implemented for convex MINLP like OA and ECP, which in turn can be used as heuristics for nonconvex MINLP problems.

.. _Kronqvist et al., 2019: https://link.springer.com/article/10.1007/s11081-018-9411-8
.. _Kesavan et al., 2004: https://link.springer.com/article/10.1007/s10107-004-0503-1
.. _MC++: https://pyomo.readthedocs.io/en/stable/contributed_packages/mcpp.html


Regularization
---------------------------------------------

As a new implementation in MindtPy, we provide a flexible regularization technique implementation. In this technique, an extra mixed-integer problem in solved in each decomposition iteration or incumbent solution of the single-tree solution methods. The extra mixed-integer program is constructed to provide a point where the NLP problem is solved closer to the feasible region described by the non-linear constraint. This approach has been proposed in [`Kronqvist et al., 2020`_] and it has shown to be efficient for highly nonlinear convex MINLP problems. In [`Kronqvist et al., 2020`_] two different regularization approaches are proposed, using an squared Euclidean norm which was proved to make the procedure equivalent to adding trust-region constraints to Outer-approximation, and a second order approximation of the Lagrangian of the problem, which showed better performance. We implement these methods, using PyomoNLP as the interface to compute the second order approximation of the Lagrangian, and extend them to consider linear norm objectives and first order approximations of the Lagrangian. Finally, we implemented an approximated second order expansion of the Lagrangian, drawing inspiration from the Sequantial Quadratic Programming (SQP) literature. The details of this implementation are included in an upcoming paper.

.. _Kronqvist et al., 2020: https://link.springer.com/article/10.1007/s10107-018-1356-3



MindtPy implementation and optional arguments
---------------------------------------------

.. warning::

   MindtPy optional arguments should be considered beta code and are
   subject to change.

.. autoclass:: pyomo.contrib.mindtpy.MindtPy.MindtPySolver
    :members:
