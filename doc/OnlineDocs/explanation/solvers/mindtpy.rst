MindtPy Solver
==============

The Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy) solver
allows users to solve Mixed-Integer Nonlinear Programs (MINLP) using decomposition algorithms.
These decomposition algorithms usually rely on the solution of Mixed-Integer Linear Programs
(MILP) and Nonlinear Programs (NLP).

The following algorithms are currently available in MindtPy:

- **Outer-Approximation (OA)** [`Duran & Grossmann, 1986`_]
- **LP/NLP based Branch-and-Bound (LP/NLP BB)** [`Quesada & Grossmann, 1992`_]
- **Extended Cutting Plane (ECP)**  [`Westerlund & Petterson, 1995`_]
- **Global Outer-Approximation (GOA)** [`Kesavan & Allgor, 2004`_]
- **Regularized Outer-Approximation (ROA)** [`Bernal & Peng, 2021`_, `Kronqvist & Bernal, 2018`_]
- **Feasibility Pump (FP)** [`Bernal & Vigerske, 2019`_, `Bonami & Cornuéjols, 2009`_]

Usage and early implementation details for MindtPy can be found in the PSE 2018 paper Bernal et al.,
(`ref <https://doi.org/10.1016/B978-0-444-64241-7.50144-0>`_,
`preprint <https://egon.cheme.cmu.edu/Papers/Bernal_Chen_MindtPy_PSE2018Paper.pdf>`_).
This solver implementation has been developed by `David Bernal <https://github.com/bernalde>`_
and `Zedong Peng <https://github.com/ZedongPeng>`_ as part of research efforts at the `Bernal Research Group 
<https://bernalde.github.io/>`_ and the `Grossmann Research Group <https://egon.cheme.cmu.edu/>`_
at Purdue University and Carnegie Mellon University.

.. _Duran & Grossmann, 1986: https://dx.doi.org/10.1007/BF02592064
.. _Westerlund & Petterson, 1995: http://dx.doi.org/10.1016/0098-1354(95)87027-X
.. _Kesavan & Allgor, 2004: https://link.springer.com/article/10.1007/s10107-004-0503-1
.. _Bernal & Peng, 2021: http://www.optimization-online.org/DB_HTML/2021/06/8452.html
.. _Kronqvist & Bernal, 2018: https://link.springer.com/article/10.1007%2Fs10107-018-1356-3
.. _Bonami & Cornuéjols, 2009: https://link.springer.com/article/10.1007/s10107-008-0212-2
.. _Bernal & Vigerske, 2019: https://www.tandfonline.com/doi/abs/10.1080/10556788.2019.1641498
.. _Kronqvist et al., 2019: https://link.springer.com/article/10.1007/s11081-018-9411-8

MINLP Formulation
-----------------

The general formulation of the mixed integer nonlinear programming (MINLP) models is as follows.

.. math::
    :nowrap:

    \begin{equation}
    \label{eq:MINLP}
    \tag{MINLP}
    \begin{aligned}
      &\min_{\mathbf{x,y}} &&f(\mathbf{x,y})\\
    & \text{s.t.} \ &&g_j(\mathbf{x,y}) \leq 0 \quad \ \forall j=1,\dots l,\\
    & &&\mathbf{A}\mathbf{x} +\mathbf{B}\mathbf{y} \leq \mathbf{b}, \\
    & &&\mathbf{x}\in {\mathbb R}^n,\ \mathbf{y} \in {\mathbb Z}^m.
    \end{aligned}
    \end{equation}

where 

- :math:`\mathbf{x}\in {\mathbb R}^n` are continuous variables,
- :math:`\mathbf{y} \in {\mathbb Z}^m` are discrete variables, 
- :math:`f, g_1, \dots, g_l` are non-linear smooth functions, 
- :math:`\mathbf{A}\mathbf{x} +\mathbf{B}\mathbf{y} \leq \mathbf{b}`` are linear constraints.

Solve Convex MINLPs
-------------------

Usage of MindtPy to solve a convex MINLP Pyomo model involves:

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model)

An example which includes the modeling approach may be found below.

.. doctest::

  Required imports
  >>> import pyomo.environ as pyo

  Create a simple model
  >>> model = pyo.ConcreteModel()

  >>> model.x = pyo.Var(bounds=(1.0,10.0),initialize=5.0)
  >>> model.y = pyo.Var(within=pyo.Binary)

  >>> model.c1 = pyo.Constraint(expr=(model.x-4.0)**2 - model.x <= 50.0*(1-model.y))
  >>> model.c2 = pyo.Constraint(expr=model.x*pyo.log(model.x)+5.0 <= 50.0*(model.y))

  >>> model.objective = pyo.Objective(expr=model.x, sense=pyo.minimize)

  Solve the model using MindtPy
  >>> pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt') # doctest: +SKIP

The solution may then be displayed by using the commands

.. code::

  >>> model.objective.display()
  >>> model.display()
  >>> model.pprint()

.. note::

   When troubleshooting, it can often be helpful to turn on verbose
   output using the ``tee`` flag.

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)

MindtPy also supports setting options for mip solvers and nlp solvers. 

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model, 
                                     strategy='OA',
                                     time_limit=3600,
                                     mip_solver='gams',
                                     mip_solver_args=dict(solver='cplex', warmstart=True),
                                     nlp_solver='ipopt',
                                     tee=True)

There are three initialization strategies in MindtPy: ``rNLP``, ``initial_binary``, ``max_binary``. In OA and GOA strategies, the default initialization strategy is ``rNLP``. In ECP strategy, the default initialization strategy is ``max_binary``.

LP/NLP Based Branch-and-Bound
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MindtPy also supports single-tree implementation of Outer-Approximation (OA) algorithm, which is known as LP/NLP based branch-and-bound algorithm originally described in [`Quesada & Grossmann, 1992`_].
The LP/NLP based branch-and-bound algorithm in MindtPy is implemented based on the LazyConstraintCallback function in commercial solvers.

.. _Quesada & Grossmann, 1992: https://www.sciencedirect.com/science/article/abs/pii/0098135492800288

.. note::

   In Pyomo, :ref:`persistent solvers <persistent_solvers>` are necessary to set or register callback functions. The single tree implementation currently only works with CPLEX and GUROBI, more exactly ``cplex_persistent`` and ``gurobi_persistent``. To use the `LazyConstraintCallback`_ function of CPLEX from Pyomo, the `CPLEX Python API`_ is required. This means both IBM ILOG CPLEX Optimization Studio and the CPLEX-Python modules should be installed on your computer. To use the `cbLazy`_ function of GUROBI from pyomo, `gurobipy`_ is required.

.. _CPLEX Python API: https://www.ibm.com/docs/en/icos/20.1.0?topic=cplex-setting-up-python-api
.. _gurobipy: https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_grbpy_the_gurobi_python.html
.. _LazyConstraintCallback: https://www.ibm.com/docs/en/icos/20.1.0?topic=classes-cplexcallbackslazyconstraintcallback
.. _cbLazy: https://www.gurobi.com/documentation/9.1/refman/py_model_cblazy.html

A usage example for LP/NLP based branch-and-bound algorithm is as follows:

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model,
  ...                                    strategy='OA',
  ...                                    mip_solver='cplex_persistent',  # or 'gurobi_persistent'
  ...                                    nlp_solver='ipopt',
  ...                                    single_tree=True)
  >>> model.objective.display()


Regularized Outer-Approximation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a new implementation in MindtPy, we provide a flexible regularization technique implementation. In this technique, an extra mixed-integer problem is solved in each decomposition iteration or incumbent solution of the single-tree solution methods. The extra mixed-integer program is constructed to provide a point where the NLP problem is solved closer to the feasible region described by the non-linear constraint. This approach has been proposed in [`Kronqvist et al., 2020`_], and it has shown to be efficient for highly non-linear convex MINLP problems. In [`Kronqvist et al., 2020`_], two different regularization approaches are proposed, using a squared Euclidean norm which was proved to make the procedure equivalent to adding a trust-region constraint to Outer-approximation, and a second-order approximation of the Lagrangian of the problem, which showed better performance. We implement these methods, using PyomoNLP as the interface to compute the second-order approximation of the Lagrangian, and extend them to consider linear norm objectives and first-order approximations of the Lagrangian. Finally, we implemented an approximated second-order expansion of the Lagrangian, drawing inspiration from the Sequential Quadratic Programming (SQP) literature. The details of this implementation are included in [`Bernal et al., 2021`_].

.. _Kronqvist et al., 2020: https://link.springer.com/article/10.1007/s10107-018-1356-3
.. _Bernal et al., 2021: http://www.optimization-online.org/DB_HTML/2021/06/8452.html

A usage example for regularized OA is as follows:

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model,
  ...                                    strategy='OA',
  ...                                    mip_solver='cplex',
  ...                                    nlp_solver='ipopt',
  ...                                    add_regularization='level_L1' 
  ...                                    # alternative regularizations
  ...                                    # 'level_L1', 'level_L2', 'level_L_infinity', 
  ...                                    # 'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'
  ...                                    )
  >>> model.objective.display()


Solution Pool Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MindtPy supports solution pool of the MILP solver, CPLEX and GUROBI. With the help of the solution, MindtPy can explore several integer combinations in one iteration. 

A usage example for OA with solution pool is as follows:

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model,
  ...                                    strategy='OA',
  ...                                    mip_solver='cplex_persistent',
  ...                                    nlp_solver='ipopt',
  ...                                    solution_pool=True,
  ...                                    num_solution_iteration=10, # default=5
  ...                                    tee=True
  ...                                    )
  >>> model.objective.display()

Feasibility Pump
^^^^^^^^^^^^^^^^

For some MINLP problems, the Outer Approximation method might have difficulty in finding a feasible solution. MindtPy provides the Feasibility Pump implementation to find feasible solutions for convex MINLPs quickly. The main idea of the Feasibility Pump is to decompose the original mixed-integer problem into two parts: integer feasibility and constraint feasibility. For convex MINLPs, a MIP is solved to obtain a solution, which satisfies the integrality constraints on `y`, but may violate some of the nonlinear constraints; next, by solving an NLP, a solution is computed that satisfies the nonlinear constraints but might again violate the integrality constraints on `y`. By minimizing the distance between these two types of solutions iteratively, a constraint and integer feasible solution can be expected. In MindtPy, the Feasibility Pump can be used both as an initialization strategy and a decomposition strategy. For details of this implementation are included in [`Bernal et al., 2017`_].

.. _Bernal et al., 2017: http://www.optimization-online.org/DB_HTML/2017/08/6171.html

A usage example for Feasibility Pump as the initialization strategy is as follows:

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model,
  ...                                    strategy='OA',
  ...                                    init_strategy='FP',
  ...                                    mip_solver='cplex',
  ...                                    nlp_solver='ipopt',
  ...                                    tee=True
  ...                                    )
  >>> model.objective.display()

A usage example for Feasibility Pump as the decomposition strategy is as follows:

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model,
  ...                                    strategy='FP',
  ...                                    mip_solver='cplex',
  ...                                    nlp_solver='ipopt',
  ...                                    tee=True
  ...                                    )
  >>> model.objective.display()



Solve Nonconvex MINLPs
----------------------


Equality Relaxation
^^^^^^^^^^^^^^^^^^^

Under certain assumptions concerning the convexity of the nonlinear functions, an equality constraint can be relaxed to be an inequality constraint. This property can be used in the MIP master problem to accumulate linear approximations(OA cuts). The sense of the equivalent inequality constraint is based on the sign of the dual values of the equality constraint. Therefore, the sense of the OA cuts for equality constraint should be determined according to both the objective sense and the sign of the dual values. In MindtPy, the dual value of the equality constraint is calculated as follows.

+--------------------+-----------------------+-------------------------+
|     constraint     | status at :math:`x_1` | dual values             |
+====================+=======================+=========================+
| :math:`g(x) \le b` | :math:`g(x_1) \le b`  |           0             |
+--------------------+-----------------------+-------------------------+
| :math:`g(x) \le b` | :math:`g(x_1) > b`    | :math:`g(x1) - b`       |
+--------------------+-----------------------+-------------------------+
| :math:`g(x) \ge b` | :math:`g(x_1) \ge b`  |           0             |
+--------------------+-----------------------+-------------------------+
| :math:`g(x) \ge b` | :math:`g(x_1) < b`    | :math:`b - g(x1)`       |
+--------------------+-----------------------+-------------------------+

Augmented Penalty
^^^^^^^^^^^^^^^^^

Augmented Penalty refers to the introduction of (non-negative) slack variables on the right hand sides of the just described inequality constraints and the modification of the objective function when assumptions concerning convexity do not hold. (From DICOPT)


Global Outer-Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Apart from the decomposition methods for convex MINLP problems [`Kronqvist et al., 2019`_], MindtPy provides an implementation of Global Outer Approximation (GOA) as described in [`Kesavan & Allgor, 2004`_], to provide optimality guaranteed for nonconvex MINLP problems. Here, the validity of the Mixed-integer Linear Programming relaxation of the original problem is guaranteed via the usage of Generalized McCormick envelopes, computed using the :ref:`interface to the MC++ package <MC++>`. The NLP subproblems, in this case, need to be solved to global optimality, which can be achieved through global NLP solvers such as `BARON`_ or `SCIP`_.

.. _BARON: https://minlp.com/baron-solver
.. _SCIP: https://www.scipopt.org/


Convergence
"""""""""""

MindtPy provides two ways to guarantee the finite convergence of the algorithm. 

- **No-good cuts**. No-good cuts(integer cuts) are added to the MILP master problem in each iteration. 
- **Tabu list**. Tabu list is only supported if the ``mip_solver`` is ``cplex_persistent`` (``gurobi_persistent`` pending). In each iteration, the explored integer combinations will be added to the `tabu_list`. When solving the next MILP problem, the MIP solver will reject the previously explored solutions in the branch and bound process through IncumbentCallback.


Bound Calculation
"""""""""""""""""

Since no-good cuts or tabu list is applied in the Global Outer-Approximation (GOA) method, the MILP master problem cannot provide a valid bound for the original problem. After the GOA method has converged, MindtPy will remove the no-good cuts or the tabu integer combinations added when and after the optimal solution has been found. Solving this problem will give us a valid bound for the original problem.


The GOA method also has a single-tree implementation with ``cplex_persistent`` and ``gurobi_persistent``. Notice that this method is more computationally expensive than the other strategies implemented for convex MINLP like OA and ECP, which can be used as heuristics for nonconvex MINLP problems.

A usage example for GOA is as follows:

.. code::

  >>> pyo.SolverFactory('mindtpy').solve(model,
  ...                                    strategy='GOA',
  ...                                    mip_solver='cplex',
  ...                                    nlp_solver='baron')
  >>> model.objective.display()



MindtPy Implementation and Optional Arguments
---------------------------------------------

.. warning::

   MindtPy optional arguments should be considered beta code and are
   subject to change.

.. autoclass:: pyomo.contrib.mindtpy.MindtPy.MindtPySolver
    :noindex:
    :members:

Get Help
--------

Ways to get help: https://github.com/Pyomo/pyomo#getting-help

Report a Bug
------------

If you find a bug in MindtPy, we will be grateful if you could

- submit an `issue`_ in Pyomo repository  
- directly contact David Bernal <dbernaln@purdue.edu> and Zedong Peng <zdpeng95@gmail.com>.

.. _issue: https://github.com/Pyomo/pyomo/issues
