==========================
Getting Started with PyROS
==========================

.. contents:: Table of Contents
   :depth: 3
   :local:


.. _pyros_installation:

Installation
============
In advance of using PyROS to solve robust optimization problems,
you will need (at least) one local nonlinear programming (NLP) solver
(e.g.,
`CONOPT <https://conopt.gams.com/>`_,
`IPOPT <https://github.com/coin-or/Ipopt>`_,
`Knitro <https://www.artelys.com/solvers/knitro/>`_)
and (at least) one global NLP solver
(e.g.,
`BARON <https://minlp.com/baron-solver>`_,
`COUENNE <https://www.coin-or.org/Couenne/>`_,
`SCIP <https://www.scipopt.org/>`_)
installed and licensed on your system.

PyROS can be installed as follows:

1. :ref:`Install Pyomo <pyomo_installation>`.
   PyROS is included in the Pyomo software package, at pyomo/contrib/pyros.
2. Install NumPy and SciPy with your preferred package manager;
   both NumPy and SciPy are required dependencies of PyROS.
   You may install NumPy and SciPy with, for example, ``conda``:

   ::

      conda install numpy scipy

   or ``pip``:

   ::

      pip install numpy scipy
3. (*Optional*) Test your installation:
   install ``pytest`` and ``parameterized``
   with your preferred package manager (as in the previous step):

   ::

      pip install pytest parameterized

   You may then run the PyROS tests as follows:

   ::

      python -c 'import os, pytest, pyomo.contrib.pyros as p; pytest.main([os.path.dirname(p.__file__)])'

   Some tests involving deterministic NLP solvers may be skipped
   if
   `IPOPT <https://github.com/coin-or/Ipopt>`_,
   `BARON <https://minlp.com/baron-solver>`_,
   or
   `SCIP <https://www.scipopt.org/>`_
   is not 
   pre-installed and licensed on your system.


Quickstart
==========
We now provide a quick overview of how to use PyROS
to solve a robust optimization problem.

Consider the nonconvex deterministic QCQP

.. math::
   :nowrap:

   \[\begin{array}{clll}
    \displaystyle \min_{\substack{x \in [-100, 100], \\ z \in [-100, 100], \\ (y_1, y_2) \in [-100, 100]^2}}
      & ~~ x^2 - y_1 z + y_2 & \\
    \displaystyle \text{s.t.}
      & ~~ xy_1 \geq 150 (q_1 + 1)^2 \\
      & ~~ x + y_2^2 \leq 600  \\
      & ~~ xz - q_2 y_1 = 2  \\
      & ~~ y_1^2 - 2y_2 = 15
   \end{array}\]

in which
:math:`x` is the sole first-stage variable,
:math:`z` is the sole second-stage variable,
:math:`y_1, y_2` are the state variables,
and :math:`q_1, q_2` are the uncertain parameters.

The uncertain parameters :math:`q_1, q_2`
each have a nominal value of 1.
We assume that :math:`q_1, q_2`
can independently deviate from their
nominal values by up to :math:`\pm 10\%`,
so that :math:`(q_1, q_2)` is constrained in value to the 
interval uncertainty set :math:`\mathcal{Q} = [0.9, 1.1]^2`.

.. note::
    Per our analysis, our selections of first-stage variables
    and second-stage variables in the present example
    satisfy our
    :ref:`assumption that the state variable values are
    uniquely defined <pyros_unique_state_vars>`.


Step 0: Import Pyomo and the PyROS Module
-----------------------------------------

In anticipation of using the PyROS solver and building the deterministic Pyomo
model:

.. _pyros_quickstart_module_imports:

.. doctest::

  >>> import pyomo.environ as pyo
  >>> import pyomo.contrib.pyros as pyros

Step 1: Define the Solver Inputs
--------------------------------

Deterministic Model
^^^^^^^^^^^^^^^^^^^

The model can be implemented as follows:

.. _pyros_quickstart_model_construct:

.. doctest::

  >>> m = pyo.ConcreteModel()
  >>> # parameters
  >>> m.q1 = pyo.Param(initialize=1, mutable=True)
  >>> m.q2 = pyo.Param(initialize=1, mutable=True)
  >>> # variables
  >>> m.x = pyo.Var(bounds=[-100, 100])
  >>> m.z = pyo.Var(bounds=[-100, 100])
  >>> m.y1 = pyo.Var(bounds=[-100, 100])
  >>> m.y2 = pyo.Var(bounds=[-100, 100])
  >>> # objective
  >>> m.obj = pyo.Objective(expr=m.x ** 2 - m.y1 * m.z + m.y2)
  >>> # constraints
  >>> m.ineq1 = pyo.Constraint(expr=m.x * m.y1 >= 150 * (m.q1 + 1) ** 2)
  >>> m.ineq2 = pyo.Constraint(expr=m.x + m.y2 ** 2 <= 600)
  >>> m.eq1 = pyo.Constraint(expr=m.x * m.z - m.y1 * m.q2 == 2)
  >>> m.eq2 = pyo.Constraint(expr=m.y1 ** 2 - 2 * m.y2 == 15)


Observe that the uncertain parameters :math:`q_1, q_2` are implemented
as mutable :class:`~pyomo.core.base.param.Param` objects.
See the 
:ref:`Uncertain parameters section of the
Solver Interface documentation <pyros_uncertain_params>`
for further guidance.


First-Stage and Second-Stage Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We take ``m.x`` to be the sole first-stage variable and ``m.z``
to be the sole second-stage variable:

.. doctest::

  >>> first_stage_variables = [m.x]
  >>> second_stage_variables = [m.z]


Uncertain Parameters
^^^^^^^^^^^^^^^^^^^^
The uncertain parameters are represented by ``m.q1`` and ``m.q2``:

.. doctest::

  >>> uncertain_params = [m.q1, m.q2]

Uncertainty Set
^^^^^^^^^^^^^^^
As previously discussed, we take the uncertainty set to be
the interval :math:`[0.9, 1.1]^2`,
which we can implement as a
:class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet` object:

.. doctest::

  >>> box_uncertainty_set = pyros.BoxSet(bounds=[(0.9, 1.1)] * 2)

Further information on PyROS uncertainty sets is presented in the
:ref:`Uncertainty Sets documentation <pyros_uncertainty_sets>`.

Subordinate NLP Solvers
^^^^^^^^^^^^^^^^^^^^^^^
We will use IPOPT as the subordinate local NLP solver
and BARON as the subordinate global NLP solver:

.. doctest::
  :skipif: not (baron_available and baron.license_is_valid() and ipopt_available)

  >>> local_solver = pyo.SolverFactory("ipopt")
  >>> global_solver = pyo.SolverFactory("baron")

.. note::

  Additional NLP optimizers can be automatically used in the event the primary
  subordinate local or global optimizer passed
  to the PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
  does not successfully solve a subproblem to an appropriate termination
  condition. These alternative solvers can be provided through the optional
  keyword arguments ``backup_local_solvers`` and ``backup_global_solvers``
  to the PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method.

In advance of using PyROS, we check that the model can be solved
to optimality with the subordinate global solver:

.. _pyros_quickstart_solve_deterministic:

.. doctest::
  :skipif: not (baron_available and baron.license_is_valid() and ipopt_available)

  >>> pyo.assert_optimal_termination(global_solver.solve(m))
  >>> deterministic_obj = pyo.value(m.obj)
  >>> print(f"Optimal deterministic objective value: {deterministic_obj:.2f}")
  Optimal deterministic objective value: 5407.94


Step 2: Solve With PyROS
------------------------
PyROS can be instantiated through the Pyomo
:class:`~pyomo.opt.base.solvers.SolverFactory`:

.. doctest::

  >>> pyros_solver = pyo.SolverFactory("pyros")

Invoke PyROS
^^^^^^^^^^^^^^^^^
We now use PyROS to solve the model to robust optimality
by invoking the :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
method of the PyROS solver object:

.. _pyros_quickstart_single-stage-problem:

.. doctest::
  :skipif: not (baron_available and baron.license_is_valid() and ipopt_available)

  >>> results_1 = pyros_solver.solve(
  ...     # required arguments
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_params,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     # optional arguments: passed directly to
  ...     #  solve to robust optimality
  ...     objective_focus="worst_case",
  ...     solve_master_globally=True,
  ... )  # doctest: +ELLIPSIS
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  Robust optimal solution identified.
  ...
  All done. Exiting PyROS.
  ==============================================================================


PyROS, by default, logs to the output console the progress of the optimization
and, upon termination, a summary of the final result.
The summary includes the iteration and solve time requirements,
the final objective function value, and the termination condition.
For further information on the output log,
see the :ref:`Solver Output Log documentation <pyros_solver_log>`.


.. note::

   PyROS, like other Pyomo solvers, accepts optional arguments
   passed indirectly through the keyword argument ``options``.
   This is discussed further in the
   :ref:`Optional Arguments section of the
   Solver Interface documentation <pyros_optional_arguments>`.
   Thus, the PyROS solver invocation in the
   :ref:`preceding code snippet <pyros_quickstart_single-stage-problem>`
   is equivalent to:

   .. code-block::

      results_1 = pyros_solver.solve(
          model=m,
          first_stage_variables=first_stage_variables,
          second_stage_variables=second_stage_variables,
          uncertain_params=uncertain_params,
          uncertainty_set=box_uncertainty_set,
          local_solver=local_solver,
          global_solver=global_solver,
          # optional arguments: passed indirectly to
          #  solve to robust optimality
          options={
              "objective_focus": "worst_case",
              "solve_master_globally": True,
          },
      )


Inspect the Results
^^^^^^^^^^^^^^^^^^^
The PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
returns a results object,
of type :class:`~pyomo.contrib.pyros.solve_data.ROSolveResults`,
that summarizes the outcome of invoking PyROS on a robust optimization problem.
By default, a printout of the results object is included at the end of the solver
output log.
Alternatively, we can display the results object ourselves using:

.. code::

   >>> print(results_1)  # output may vary  # doctest: +SKIP
   Termination stats:
    Iterations            : 3
    Solve time (wall s)   : 0.917
    Final objective value : 9.6616e+03
    Termination condition : pyrosTerminationCondition.robust_optimal


We can also query the results object's individual attributes:

.. code::

   >>> results_1.iterations  # total number of iterations
   3
   >>> results_1.time  # total wallclock time; may vary # doctest: +SKIP
   0.839
   >>> results_1.final_objective_value  # final objective value; may vary # doctest: +ELLIPSIS
   9661.6...
   >>> results_1.pyros_termination_condition  # termination condition
   <pyrosTerminationCondition.robust_optimal: 1>

Since PyROS has successfully solved our problem,
the final solution has been automatically loaded to the model.
We can inspect the resulting state of the model
by invoking, for example, ``m.display()`` or ``m.pprint()``.

For a general discussion of the PyROS solver outputs,
see the
:ref:`Overview of Outputs section of the
Solver Interface documentation <pyros_solver_outputs>`.


Try Higher-Order Decision Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyROS uses polynomial decision rules to approximate the adjustability
of the second-stage variables to the uncertain parameters.
The degree of the decision rule polynomials is
specified through the optional keyword argument
``decision_rule_order`` to the PyROS
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method.
By default, ``decision_rule_order`` is set to 0,
so that static decision rules are used.
Increasing the decision rule order
may yield a solution with better quality:

.. _pyros_quickstart_example-two-stg:

.. doctest::
  :skipif: not (baron_available and baron.license_is_valid() and ipopt_available)

  >>> results_2 = pyros_solver.solve(
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_params,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     objective_focus="worst_case",
  ...     solve_master_globally=True,
  ...     decision_rule_order=1,  # use affine decision rules
  ... )  # doctest: +ELLIPSIS
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  Robust optimal solution identified.
  ...
  All done. Exiting PyROS.
  ==============================================================================


Inspecting the results:

.. code::

   >>> print(results_2)  # output may vary  # doctest: +SKIP
   Termination stats:
    Iterations            : 5
    Solve time (wall s)   : 1.956
    Final objective value : 6.5403e+03
    Termination condition : pyrosTerminationCondition.robust_optimal


Notice that when we switch from optimizing over static decision rules
to optimizing over affine decision rules,
there is a ~32% decrease in the final objective
value, albeit at some additional computational expense.


Analyzing the Price of Robustness
---------------------------------
In conjunction with standard Pyomo control flow tools,
PyROS facilitates an analysis of the "price of robustness",
which we define to be the increase in the robust optimal objective value
relative to the deterministically optimal objective value.

Let us, for example, consider optimizing robustly against
an interval uncertainty set :math:`[1 - p, 1 + p]^2`,
where :math:`p` is the half-length of the interval.
We can optimize against intervals of increasing half-length :math:`p`
by iterating over select values for :math:`p` in a ``for`` loop,
and in each iteration, solving a robust optimization problem
subject to a corresponding
:class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet` instance:

.. doctest::
  :skipif: not (baron_available and baron.license_is_valid() and ipopt_available)

  >>> results_dict = dict()
  >>> for half_length in [0.0, 0.1, 0.2, 0.3, 0.4]:
  ...     print(f"Solving problem for {half_length=}:")
  ...     results_dict[half_length] = pyros_solver.solve(
  ...         model=m,
  ...         first_stage_variables=first_stage_variables,
  ...         second_stage_variables=second_stage_variables,
  ...         uncertain_params=uncertain_params,
  ...         uncertainty_set=pyros.BoxSet(
  ...             bounds=[(1 - half_length, 1 + half_length)] * 2
  ...         ),
  ...         local_solver=local_solver,
  ...         global_solver=global_solver,
  ...         objective_focus="worst_case",
  ...         solve_master_globally=True,
  ...         decision_rule_order=1,
  ...     )  # doctest: +ELLIPSIS
  ...
  Solving problem for half_length=0.0:
  ...
  Solving problem for half_length=0.1:
  ...
  Solving problem for half_length=0.2:
  ...
  Solving problem for half_length=0.3:
  ...
  Solving problem for half_length=0.4:
  ...
  All done. Exiting PyROS.
  ==============================================================================

Using the :py:obj:`dict` populated in the loop,
and the 
:ref:`previously evaluated deterministically optimal
objective value <pyros_quickstart_solve_deterministic>`,
we can print a tabular summary of the results:

.. doctest::
   :skipif: not (baron_available and baron.license_is_valid() and ipopt_available)

   >>> for idx, (half_length, res) in enumerate(results_dict.items()):
   ...     if idx == 0:
   ...         # print table header
   ...         print("=" * 71)
   ...         print(
   ...             f"{'Half-Length':15s}"
   ...             f"{'Termination Cond.':21s}"
   ...             f"{'Objective Value':18s}"
   ...             f"{'Price of Rob. (%)':17s}"
   ...         )
   ...         print("-" * 71)
   ...     # print table row
   ...     obj_value, percent_obj_increase = float("nan"), float("nan")
   ...     is_robust_optimal = (
   ...         res.pyros_termination_condition
   ...         == pyros.pyrosTerminationCondition.robust_optimal
   ...     )
   ...     is_robust_infeasible = (
   ...         res.pyros_termination_condition
   ...         == pyros.pyrosTerminationCondition.robust_infeasible
   ...     )
   ...     if is_robust_optimal:
   ...         # compute the price of robustness
   ...         obj_value = res.final_objective_value
   ...         price_of_robustness = (
   ...             (res.final_objective_value - deterministic_obj)
   ...             / deterministic_obj
   ...         )
   ...     elif is_robust_infeasible:
   ...         # infinite objective
   ...         obj_value, price_of_robustness = float("inf"), float("inf")
   ...     print(
   ...         f"{half_length:<15.1f}"
   ...         f"{res.pyros_termination_condition.name:21s}"
   ...         f"{obj_value:<18.2f}"
   ...         f"{100 * price_of_robustness:<.2f}"
   ...     )
   ...     print("-" * 71)
   ...
   =======================================================================
   Half-Length    Termination Cond.    Objective Value   Price of Rob. (%)
   -----------------------------------------------------------------------
   0.0            robust_optimal       5407.94           -0.00
   -----------------------------------------------------------------------
   0.1            robust_optimal       6540.31           20.94
   -----------------------------------------------------------------------
   0.2            robust_optimal       7838.50           44.94
   -----------------------------------------------------------------------
   0.3            robust_optimal       9316.88           72.28
   -----------------------------------------------------------------------
   0.4            robust_infeasible    inf               inf
   -----------------------------------------------------------------------


The table shows the response of the PyROS termination condition,
final objective value, and price of robustness
to the half-length :math:`p`.
Observe that:

* The optimal objective value for the interval of half-length
  :math:`p=0` is equal to the optimal deterministic objective value
* The objective value (and thus, the price of robustness)
  increases with the half-length
* For large enough half-length (:math:`p=0.4`), the problem
  is robust infeasible

Therefore, this example clearly illustrates the potential
impact of the uncertainty set size on the robust optimal
objective function value
and the ease of analyzing the price of robustness
for a given optimization problem under uncertainty.


Beyond the Basics
=================
A more in-depth guide to incorporating PyROS into a
Pyomo optimization workflow is given
in the :ref:`Usage Tutorial <pyros_tutorial>`.
