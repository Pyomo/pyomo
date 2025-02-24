.. _pyros_installation:

Getting Started with PyROS
==========================

.. contents:: Table of Contents
   :depth: 2
   :local:


Installation
------------
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

   Some tests involving solvers may fail or be skipped,
   depending on the solver distributions (e.g., Ipopt, BARON, SCIP)
   that you have pre-installed and licensed on your system.

Usage Tutorial
--------------
In this tutorial, we will solve robust optimization problems with PyROS.
The problems are derived from the deterministic model *hydro*,
a QCQP taken from the
`GAMS Model Library <https://www.gams.com/latest/gamslib_ml/libhtml/>`_.
We have converted the
`GAMS implementation of hydro <https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_hydro.html>`_
to Pyomo format using the
`GAMS CONVERT utility <https://www.gams.com/latest/docs/S_CONVERT.html>`_.

The model *hydro* features 31 variables,
13 of which are considered independent variables,
and the remaining 18 of which are considered state variables.
Moreover, there are
6 linear inequality constraints,
12 linear equality constraints,
6 non-linear (quadratic) equality constraints,
and a quadratic objective.
**All variables of the model are continuous.**

.. note::
   PyROS does not support models with discrete decision variables.

.. note::
    Per our analysis, the model *hydro* satisfies the requirement that
    each value of :math:`\left(x, z, q \right)` maps to a unique
    value of :math:`y`, which, in accordance with
    :ref:`our assumption on the equality constraints <unique-mapping>`,
    indicates a proper partitioning of the model variables
    into (first-stage and second-stage) degrees of freedom and
    state variables.

We have extended this model by converting one objective coefficient,
two constraint coefficients, and one constraint right-hand side
into :class:`~pyomo.core.base.param.Param` objects
so that they can be considered uncertain later on.


Step 0: Import Pyomo and the PyROS Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In anticipation of using the PyROS solver and building the deterministic Pyomo
model:

.. _pyros_module_imports:

.. doctest::

  >>> import pyomo.environ as pyo
  >>> import pyomo.contrib.pyros as pyros

Step 1: Define the Deterministic Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The deterministic Pyomo model for *hydro* is constructed as follows.
We first instantiate a Pyomo model object:

.. _pyros_model_construct:

.. doctest::

  >>> m = pyo.ConcreteModel()
  >>> m.name = "hydro"

Some constants of the model are later considered uncertain.
These are represented by mutable :class:`~pyomo.core.base.param.Param` objects:

.. doctest::

  >>> nominal_values = {0: 82.8*0.0016, 1: 4.97, 2: 4.97, 3: 1800}
  >>> m.q = pyo.Param(
  ...     list(nominal_values),
  ...     initialize=nominal_values,
  ...     mutable=True,
  ... )

.. note::
    Primitive data (Python literals) that have been hard-coded within a
    deterministic model (:class:`~pyomo.core.base.PyomoModel.ConcreteModel`)
    cannot be later considered uncertain,
    unless they are first converted to Pyomo
    :class:`~pyomo.core.base.param.Param` instances declared on the
    :class:`~pyomo.core.base.PyomoModel.ConcreteModel` object.
    Furthermore, any :class:`~pyomo.core.base.param.Param`
    object that is to be later considered uncertain must be instantiated
    with the argument ``mutable=True``.

.. note::
    If specifying/modifying the ``mutable`` argument in the
    :class:`~pyomo.core.base.param.Param` declarations
    of your deterministic model source code
    is not straightforward in your context, then
    you may consider adding **after** the
    :ref:`Pyomo/PyROS module imports <pyros_module_imports>`
    but **before**
    :ref:`instantiating the model object <pyros_model_construct>`
    the statement:

    .. code::

       pyo.Param.DefaultMutable = True

    For all :class:`~pyomo.core.base.param.Param`
    objects declared after this statement,
    the attribute ``mutable`` is set to True by default.
    Hence, non-mutable :class:`~pyomo.core.base.param.Param`
    objects are now declared by explicitly passing the argument
    ``mutable=False`` to the :class:`~pyomo.core.base.param.Param`
    constructor.


Finally, we declare the decision variables, objective, and constraints:

.. doctest::

  >>> # declare variables
  >>> m.x1 = pyo.Var(within=pyo.Reals, bounds=(150, 1500), initialize=150)
  >>> m.x2 = pyo.Var(within=pyo.Reals, bounds=(150, 1500), initialize=150)
  >>> m.x3 = pyo.Var(within=pyo.Reals, bounds=(150, 1500), initialize=150)
  >>> m.x4 = pyo.Var(within=pyo.Reals, bounds=(150, 1500), initialize=150)
  >>> m.x5 = pyo.Var(within=pyo.Reals, bounds=(150, 1500), initialize=150)
  >>> m.x6 = pyo.Var(within=pyo.Reals, bounds=(150, 1500), initialize=150)
  >>> m.x7 = pyo.Var(within=pyo.Reals, bounds=(0, 1000), initialize=0)
  >>> m.x8 = pyo.Var(within=pyo.Reals, bounds=(0, 1000), initialize=0)
  >>> m.x9 = pyo.Var(within=pyo.Reals, bounds=(0, 1000), initialize=0)
  >>> m.x10 = pyo.Var(within=pyo.Reals, bounds=(0, 1000), initialize=0)
  >>> m.x11 = pyo.Var(within=pyo.Reals, bounds=(0, 1000), initialize=0)
  >>> m.x12 = pyo.Var(within=pyo.Reals, bounds=(0, 1000), initialize=0)
  >>> m.x13 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x14 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x15 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x16 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x17 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x18 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x19 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x20 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x21 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x22 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x23 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x24 = pyo.Var(within=pyo.Reals, bounds=(0, None), initialize=0)
  >>> m.x25 = pyo.Var(within=pyo.Reals, bounds=(100000, 100000), initialize=100000)
  >>> m.x26 = pyo.Var(within=pyo.Reals, bounds=(60000, 120000), initialize=60000)
  >>> m.x27 = pyo.Var(within=pyo.Reals, bounds=(60000, 120000), initialize=60000)
  >>> m.x28 = pyo.Var(within=pyo.Reals, bounds=(60000, 120000), initialize=60000)
  >>> m.x29 = pyo.Var(within=pyo.Reals, bounds=(60000, 120000), initialize=60000)
  >>> m.x30 = pyo.Var(within=pyo.Reals, bounds=(60000, 120000), initialize=60000)
  >>> m.x31 = pyo.Var(within=pyo.Reals, bounds=(60000, 120000), initialize=60000)
  >>>
  >>> # declare objective
  >>> m.obj = pyo.Objective(
  ...     expr=(
  ...         m.q[0]*m.x1**2 + 82.8*8*m.x1 + 82.8*0.0016*m.x2**2
  ...         + 82.8*82.8*8*m.x2 + 82.8*0.0016*m.x3**2 + 82.8*8*m.x3
  ...         + 82.8*0.0016*m.x4**2 + 82.8*8*m.x4 + 82.8*0.0016*m.x5**2
  ...         + 82.8*8*m.x5 + 82.8*0.0016*m.x6**2 + 82.8*8*m.x6 + 248400
  ...    ),
  ...    sense=pyo.minimize,
  ... )
  >>> 
  >>> # declare constraints
  >>> m.c2 = pyo.Constraint(expr=-m.x1 - m.x7 + m.x13 + 1200<= 0)
  >>> m.c3 = pyo.Constraint(expr=-m.x2 - m.x8 + m.x14 + 1500 <= 0)
  >>> m.c4 = pyo.Constraint(expr=-m.x3 - m.x9 + m.x15 + 1100 <= 0)
  >>> m.c5 = pyo.Constraint(expr=-m.x4 - m.x10 + m.x16 + m.q[3] <= 0)
  >>> m.c6 = pyo.Constraint(expr=-m.x5 - m.x11 + m.x17 + 950 <= 0)
  >>> m.c7 = pyo.Constraint(expr=-m.x6 - m.x12 + m.x18 + 1300 <= 0)
  >>> m.c8 = pyo.Constraint(expr=12*m.x19 - m.x25 + m.x26 == 24000)
  >>> m.c9 = pyo.Constraint(expr=12*m.x20 - m.x26 + m.x27 == 24000)
  >>> m.c10 = pyo.Constraint(expr=12*m.x21 - m.x27 + m.x28 == 24000)
  >>> m.c11 = pyo.Constraint(expr=12*m.x22 - m.x28 + m.x29 == 24000)
  >>> m.c12 = pyo.Constraint(expr=12*m.x23 - m.x29 + m.x30 == 24000)
  >>> m.c13 = pyo.Constraint(expr=12*m.x24 - m.x30 + m.x31 == 24000)
  >>> m.c14 = pyo.Constraint(expr=-8e-5*m.x7**2 + m.x13 == 0)
  >>> m.c15 = pyo.Constraint(expr=-8e-5*m.x8**2 + m.x14 == 0)
  >>> m.c16 = pyo.Constraint(expr=-8e-5*m.x9**2 + m.x15 == 0)
  >>> m.c17 = pyo.Constraint(expr=-8e-5*m.x10**2 + m.x16 == 0)
  >>> m.c18 = pyo.Constraint(expr=-8e-5*m.x11**2 + m.x17 == 0)
  >>> m.c19 = pyo.Constraint(expr=-8e-5*m.x12**2 + m.x18 == 0)
  >>> m.c20 = pyo.Constraint(expr=-4.97*m.x7 + m.x19 == 330)
  >>> m.c21 = pyo.Constraint(expr=-m.q[1]*m.x8 + m.x20 == 330)
  >>> m.c22 = pyo.Constraint(expr=-4.97*m.x9 + m.x21 == 330)
  >>> m.c23 = pyo.Constraint(expr=-4.97*m.x10 + m.x22 == 330)
  >>> m.c24 = pyo.Constraint(expr=-m.q[2]*m.x11 + m.x23 == 330)
  >>> m.c25 = pyo.Constraint(expr=-4.97*m.x12 + m.x24 == 330)


Before moving on, we check that the model can be solved to optimality
with a deterministic nonlinear programming (NLP) solver.
We have elected to use BARON as the solver:

.. _pyros_solve_deterministic:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> baron = pyo.SolverFactory("baron")
  >>> pyo.assert_optimal_termination(baron.solve(m))
  >>> deterministic_obj = pyo.value(m.obj)
  >>> print("Optimal deterministic objective value: {deterministic_obj:.4e}")
  Optimal deterministic objective value: 3.5838e+07


Step 2: Define the Uncertainty Quantification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first collect the components of our model that represent the
uncertain parameters.
In this example, we assume that the quantities
represented by ``m.q[0]``, ``m.q[1]``, ``m.q[2]``, and ``m.q[3]``
are the uncertain parameters.
Since these objects comprise the mutable :class:`~pyomo.core.base.param.Param`
object ``m.q``, we can conveniently specify:

.. doctest::

  >>> uncertain_params = m.q

Equivalently, we may instead set ``uncertain_params`` to
one of the following:

* ``[m.q]``
* ``[m.q[0], m.q[1], m.q[2], m.q[3]]``
* ``list(m.q.values())``

.. note::
    1. Any :class:`~pyomo.core.base.param.Param` object that
       represents an uncertain parameter must be instantiated
       with the constructor argument ``mutable=True``.
    2. Uncertain parameters can also be represented by
       :class:`~pyomo.core.base.var.Var`
       objects declared on the deterministic model.
       Prior to invoking PyROS,
       all such :class:`~pyomo.core.base.var.Var` objects should be fixed.


PyROS requires an uncertainty set against which to robustly
optimize the model.
The goal is to identify a solution to the model that remains feasible
subject to any uncertain parameter realization located within
the uncertainty set.
In PyROS, an uncertainty set is represented by
an instance of a subclass of the
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet` class.

In the present example,
let us assume that each uncertain parameter can
independently of the other uncertain parameters
deviate from the parameter's nominal value by up to :math:`\pm 15\%`.
Then the parameter values are constrained to a box region,
which we can implement as an instance of the
:class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet` subclass:

.. doctest::

  >>> relative_deviation = 0.15
  >>> box_uncertainty_set = pyros.BoxSet(bounds=[
  ...     (val * (1 - relative_deviation), val * (1 + relative_deviation))
  ...     for val in nominal_values.values()
  ... ])

Further information on PyROS uncertainty sets is presented in the
:ref:`Uncertainty Sets section <pyros_uncertainty_sets>`.


Step 3: Solve With PyROS
^^^^^^^^^^^^^^^^^^^^^^^^^^
PyROS can be instantiated through the Pyomo
:class:`~pyomo.opt.base.solvers.SolverFactory`:

.. doctest::

  >>> pyros_solver = pyo.SolverFactory("pyros")

PyROS requires the user to supply one subordinate local NLP optimizer
and one subordinate global NLP optimizer for solving subproblems.
For convenience, we shall have PyROS use
:ref:`the previously instantiated BARON solver <pyros_solve_deterministic>`
as both the subordinate local and global NLP solvers:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> local_solver = baron
  >>> global_solver = baron

.. note::
    Additional NLP optimizers can be automatically used in the event the primary
    subordinate local or global optimizer passed
    to the PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
    does not successfully solve a subproblem to an appropriate termination
    condition. These alternative solvers are provided through the optional
    keyword arguments ``backup_local_solvers`` and ``backup_global_solvers``.

The final step in solving a model with PyROS is to construct the
remaining required inputs, namely
``first_stage_variables`` and ``second_stage_variables``.
Below, we present two separate cases.


A Single-Stage Problem
"""""""""""""""""""""""""
We can use PyROS to solve a single-stage robust optimization problem,
in which all independent variables are designated to be first-stage.
In the present example, the independent variables are
taken to be ``m.x1`` through ``m.x6``, ``m.x19`` through ``m.x24``, and ``m.x31``.
So our variable designation is as follows:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> first_stage_variables = [
  ...     m.x1, m.x2, m.x3, m.x4, m.x5, m.x6,
  ...     m.x19, m.x20, m.x21, m.x22, m.x23, m.x24, m.x31,
  ... ]
  >>> second_stage_variables = []

The single-stage problem can now be solved
to robust optimality
by invoking the :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
method of the PyROS solver object, as follows:

.. _single-stage-problem:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> results_1 = pyros_solver.solve(
  ...     # required arguments
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_params,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     # optional arguments: solve to robust optimality
  ...     objective_focus=pyros.ObjectiveType.worst_case,
  ...     solve_master_globally=True,
  ... )
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  ------------------------------------------------------------------------------
  Robust optimal solution identified.
  ...
  Termination stats:
   Iterations            : 6
   Solve time (wall s)   : 2.841
   Final objective value : 4.8367e+07
   Termination condition : pyrosTerminationCondition.robust_optimal
  ------------------------------------------------------------------------------
  All done. Exiting PyROS.
  ==============================================================================

PyROS (by default) logs to the output console the progress of the optimization
and, upon termination, a summary of the final result.
The summary includes the iteration and solve time requirements,
the final objective function value, and the termination condition.
For further information on the output log,
see the :ref:`Solver Output Log section <pyros_solver_log>`.

A Two-Stage Problem
""""""""""""""""""""""
Let us now assume that some of the independent variables are second-stage:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> first_stage_variables = [m.x5, m.x6, m.x19, m.x22, m.x23, m.x24, m.x31]
  >>> second_stage_variables = [m.x1, m.x2, m.x3, m.x4, m.x20, m.x21]


PyROS uses polynomial decision rules to approximate the adjustability
of the second-stage variables to the uncertain parameters.
The degree of the decision rule polynomials is
specified through the optional keyword argument
``decision_rule_order`` to the PyROS
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method.
In this example, we elect to use affine decision rules by
specifying ``decision_rule_order=1``.
Thus, we can solve the resulting two-stage problem 
:ref:`to robust optimality <pyros_robust_optimality_args>`
as follows:

.. _example-two-stg:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> results_2 = pyros_solver.solve(
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_params,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     objective_focus=pyros.ObjectiveType.worst_case,
  ...     solve_master_globally=True,
  ...     decision_rule_order=1,  # use affine decision rules
  ... )
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  ------------------------------------------------------------------------------
  Robust optimal solution identified.
  ...
  Termination stats:
   Iterations            : 5
   Solve time (wall s)   : 6.336
   Final objective value : 3.6285e+07
   Termination condition : pyrosTerminationCondition.robust_optimal
  ------------------------------------------------------------------------------
  All done. Exiting PyROS.
  ==============================================================================


Specifying Arguments Indirectly Through ``options``
"""""""""""""""""""""""""""""""""""""""""""""""""""
Like other Pyomo solver interface methods,
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
provides support for specifying optional arguments indirectly by passing
a keyword argument ``options``, for which the value must be a :class:`dict`
mapping names of optional arguments to
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
to their desired values.
For example, the ``solve()`` statement in the
:ref:`two-stage problem snippet <example-two-stg>`
could have been equivalently written as:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> results_2 = pyros_solver.solve(
  ...     # required arguments
  ...     model=m,
  ...     first_stage_variables=first_stage_variables,
  ...     second_stage_variables=second_stage_variables,
  ...     uncertain_params=uncertain_params,
  ...     uncertainty_set=box_uncertainty_set,
  ...     local_solver=local_solver,
  ...     global_solver=global_solver,
  ...     # optional arguments: passed indirectly
  ...     options={
  ...         "objective_focus": pyros.ObjectiveType.worst_case,
  ...         "solve_master_globally": True,
  ...         "decision_rule_order": 1,
  ...     },
  ... )
  ==============================================================================
  PyROS: The Pyomo Robust Optimization Solver...
  ...
  ------------------------------------------------------------------------------
  Robust optimal solution identified.
  ------------------------------------------------------------------------------
  ...
  Termination stats:
   Iterations            : 5
   Solve time (wall s)   : 6.336
   Final objective value : 3.6285e+07
   Termination condition : pyrosTerminationCondition.robust_optimal
  ------------------------------------------------------------------------------
  All done. Exiting PyROS.
  ==============================================================================


In the event an argument is passed directly
by position or keyword, *and* indirectly through ``options``,
the value passed directly takes precedence over the value
passed through ``options``.

.. warning::

   All required arguments to the PyROS
   :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
   must be passed directly by position or keyword,
   or else an exception is raised.
   Required arguments passed indirectly through the ``options``
   setting are ignored.


Step 4: Check the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
returns a results object,
of type :class:`~pyomo.contrib.pyros.solve_data.ROSolveResults`,
that summarizes the outcome of invoking PyROS on a robust optimization problem.
By default, a printout of the results object is included at the end of the solver
output log.
Alternatively, we can display the results object ourselves using:

.. code::

   >>> print(results_2)
   Termination stats:
    Iterations            : 5
    Solve time (wall s)   : 6.336
    Final objective value : 3.6285e+07
    Termination condition : pyrosTerminationCondition.robust_optimal

We can also query the results object's individual attributes:

.. code::

   >>> results_2.iterations  # total number of iterations
   5
   >>> results_2.time  # total wall-clock seconds; may vary
   6.336
   >>> results_2.final_objective_value  # final objective value; may vary
   36285242.22224089
   >>> results_2.pyros_termination_condition  # termination condition
   pyrosTerminationCondition.robust_optimal


The ``pyros_termination_condition`` attribute of the resuls object
is a member of the
:class:`~pyomo.contrib.pyros.util.pyrosTerminationCondition` enumeration.

.. _pyros_robust_optimality_args:

.. note::

    When the PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
    has successfully solved a given robust optimization problem,
    the :attr:`~pyomo.contrib.pyros.solve_data.ROSolveResults.pyros_termination_condition`
    attribute of the returned
    :class:`~pyomo.contrib.pyros.solve_data.ROSolveResults`
    object is set to
    :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_optimal`
    only if:

    1. Master problems are solved to global optimality
       (by specifying ``solve_master_globally=True``)
    2. A worst-case objective focus is chosen
       (by specifying ``objective_focus=pyros.ObjectiveType.worst_case``)

    Otherwise, the termination condition is set to
    :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_feasible`.

.. note::

    The reported objective and variable values
    depend on the value of the option ``objective_focus``:

    * If ``objective_focus=pyros.ObjectiveType.nominal``,
      then the objective, second-stage variables, and
      state variables are evaluated at
      the nominal uncertain parameter realization.
    * If ``objective_focus=pyros.ObjectiveType.worst_case``,
      then the objective, second-stage variables, and
      state variables are evaluated at
      the worst-case uncertain parameter realization.


We expect that adding second-stage recourse to the
single-stage *hydro* problem results in
a reduction in the robust optimal objective value.
To confirm our expectation, the final objectives can be compared as follows:

.. doctest::
  :skipif: not (baron.available() and baron.license_is_valid())

  >>> single_stage_final_objective = pyo.value(results_1.final_objective_value)
  >>> two_stage_final_objective = pyo.value(results_2.final_objective_value)
  >>> relative_obj_decrease = (
  ...     (single_stage_final_objective - two_stage_final_objective)
  ...     / single_stage_final_objective
  ... )
  >>> print(
  ...    "Percentage decrease (relative to single-stage problem objective): "
  ...    f"{100 * relative_obj_decrease:.2f}"
  ... )
  Percentage decrease (relative to single-stage problem objective): 24.98


Our check confirms that there is a ~25% decrease in the final objective
value when switching from a static decision rule
(no second-stage recourse) to an affine decision rule.

We can also inspect the state of the model after the solution
has been loaded by invoking ``m.display()`` or ``m.pprint()``.

.. note::

   PyROS loads the final solution to the deterministic model only if:

   1. The argument ``load_solution=True`` has been passed to PyROS
      (occurs by default)
   2. The termination condition is either
      :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_optimal`
      or 
      :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_feasible`

   Otherwise, the final solution is lost.


Analyzing the Price of Robustness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyROS facilitates an analysis of the "price of robustness",
which we define to be the increase in the robust optimal objective value
relative to the deterministically optimal objective value
Let us, for example, consider optimizing robustly against a
box uncertainty set centered on the nominal realization
of the uncertain parameters
and parameterized by a value :math:`p \geq 0`
specifying the half-lengths of the box relative to the nominal realization.
Then the box set is defined by:

.. math::

   \{q \in \mathbb{R}^4 \,|\, (1 - p)q^\text{nom} \leq q \leq (1 + p)q^\text{nom} \}

in which :math:`q^\text{nom}` denotes the nominal realization.
We can optimize against box sets of increasing
normalized half-length :math:`p`
by constructing a corresponding
:class:`~pyomo.contrib.pyros.uncertainty_sets.BoxSet`
instance and invoking the
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
in a for-loop:

.. code::

  >>> results_dict = dict()
  >>> for half_length in [0.0, 0.1, 0.2, 0.3, 0.4]:
  ...     print(f"Solving problem for {relative_deviation=}:")
  ...     box_uncertainty_set = pyros.BoxSet(bounds=[
  ...         (val * (1 - half_length), val * (1 + half_length))
  ...         for val in nominal_values.values()
  ...     ])
  ...     results_dict[half_length] = pyros_solver.solve(
  ...         model=m,
  ...         first_stage_variables=first_stage_variables,
  ...         second_stage_variables=second_stage_variables,
  ...         uncertain_params=uncertain_params,
  ...         uncertainty_set=box_uncertainty_set,
  ...         local_solver=local_solver,
  ...         global_solver=global_solver,
  ...         objective_focus=pyros.ObjectiveType.worst_case,
  ...         solve_master_globally=True,
  ...         decision_rule_order=1,
  ...     )
  >>> print("All done.")
  Solving problem for relative_deviation=0.0:
  ...
  Solving problem for relative_deviation=0.1:
  ...
  Solving problem for relative_deviation=0.2:
  ...
  Solving problem for relative_deviation=0.3:
  ...
  Solving problem for relative_deviation=0.4
  ...
  All done.

Using the :py:obj:`dict` populated in the loop,
and the 
:ref:`previously evaluated deterministically optimal objective value <pyros_solve_deterministic>`,
we can print a tabular summary of the results:

.. code::

   >>> print("=" * 80)
   >>> print(
   ...     f"{'Relative Half-Len.':20s}",
   ...     f"{'Termination Cond.':20s}",
   ...     f"{'Objective Value':20s}",
   ...     f"{'Price of Rob. (%)':20s}",
   ... )
   >>> print("-" * 80)
   >>> for half_length, res in results_dict.items():
   ...     obj_value, percent_obj_increase = float("nan"), float("nan")
   ...     is_robust_optimal = (
   ...         res.pyros_termination_condition
   ...         == pyros.pyrosTerminationCondition.robust_optimal
   ...     )
   ...     if is_robust_optimal:
   ...         obj_value = res.final_objective_value
   ...         price_of_robustness = (
   ...             (res.final_objective_value - deterministic_obj)
   ...             / deterministic_obj
   ...         )
   ...     print(
   ...         f"{deviation:<20.1f}",
   ...         f"{res.pyros_termination_condition.name:20s}",
   ...         f"{obj_value:<20.4e}",
   ...         f"{100 * price_of_robustness:<20.2f}",
   ...     )
   >>> print("=" * 80)
   ====================================================================================
   Relative Half-Len.   Termination Cond.    Objective Value      Price of Rob. (%)
   ------------------------------------------------------------------------------------
   0.0                  robust_optimal       3.5838e+07           0.00               
   0.1                  robust_optimal       3.6134e+07           0.83                
   0.2                  robust_optimal       3.6437e+07           1.67                
   0.3                  robust_optimal       4.3478e+07           21.32               
   0.4                  robust_infeasible    nan                  nan
   ====================================================================================


The table shows the response of the PyROS termination condition,
final objective value, and price of robustness
to the relative half-length :math:`p`.
Observe that:

* The optimal objective value for the box set of relative half-length
  :math:`p=0` is equal to the optimal deterministic objective value
* The objective value (and thus, the price of robustness)
  increases with the half-length
* For large enough half-length (:math:`p=0.4`) the problem
  is robust infeasible

Therefore, this example clearly illustrates the potential
impact of the uncertainty set size on the robust optimal
objective function value
and the ease of analyzing the price of robustness
for a given optimization problem under uncertainty.
