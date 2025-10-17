.. _pyros_solver_interface:

======================
PyROS Solver Interface
======================

.. contents:: Table of Contents
   :depth: 2
   :local:

Instantiation
=============

The PyROS solver is invoked through the
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
of an instance of the PyROS solver class, which can be 
instantiated as follows:

.. code-block::

  import pyomo.environ as pyo
  import pyomo.contrib.pyros as pyros  # register the PyROS solver
  pyros_solver = pyo.SolverFactory("pyros")


Overview of Inputs
==================

Deterministic Model
-------------------
PyROS is designed to operate on a single-objective deterministic model
(implemented as a :class:`~pyomo.core.base.PyomoModel.ConcreteModel`),
from which the robust optimization counterpart is automatically inferred.
All variables of the model should be continuous, as
mixed-integer problems are not supported.

First-Stage and Second-Stage Variables
--------------------------------------
A model may have either first-stage variables,
second-stage variables, or both.
PyROS automatically considers all other variables participating
in the active model components to be state variables.
Further, PyROS assumes that the state variables are
:ref:`uniquely defined by the equality constraints <pyros_unique_state_vars>`.


.. _pyros_uncertain_params:

Uncertain Parameters
--------------------
Uncertain parameters can be represented by either
mutable :class:`~pyomo.core.base.param.Param`
or fixed :class:`~pyomo.core.base.var.Var` objects.
Uncertain parameters *cannot* be directly
represented by Python literals that have been hard-coded into the
deterministic model.

A :class:`~pyomo.core.base.param.Param` object can be made mutable
at construction by passing the argument ``mutable=True`` to the
:class:`~pyomo.core.base.param.Param` constructor.
If specifying/modifying the ``mutable`` argument
is not straightforward in your context,
then add the following lines of code to your script
before setting up your deterministic model:


.. code-block::

   import pyomo.environ as pyo
   pyo.Param.DefaultMutable = True

All :class:`~pyomo.core.base.param.Param` objects declared
after the preceding code statements will be made mutable by default.


Uncertainty Set
---------------
The uncertainty set is represented by an
:class:`~pyomo.contrib.pyros.uncertainty_sets.UncertaintySet`
object.
See the :ref:`Uncertainty Sets documentation <pyros_uncertainty_sets>`
for more information.

Subordinate NLP Solvers
-----------------------
PyROS requires at least one subordinate
local nonlinear programming (NLP) solver (e.g., Ipopt or CONOPT)
and subordinate global NLP solver (e.g., BARON or SCIP)
to solve subproblems.

.. note::

   In advance of invoking the PyROS solver,
   check that your deterministic model can be solved
   to optimality by either your subordinate local or global
   NLP solver.


.. _pyros_optional_arguments:

Optional Arguments
------------------
The optional arguments are enumerated in the documentation of the
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method.

Like other Pyomo solver interface methods,
the PyROS
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
method
accepts the keyword argument ``options``,
which must be a :class:`dict`
mapping names of optional arguments to
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
to their desired values.
If an argument is passed directly by keyword and
indirectly through ``options``,
then the value passed directly takes precedence over the
value passed through ``options``.

.. warning::

   All required arguments to the PyROS
   :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
   must be passed directly by position or keyword,
   or else an exception is raised.
   Required arguments passed indirectly through the ``options``
   setting are ignored.


Separation Priority Ordering 
----------------------------
The PyROS solver supports custom prioritization of
the separation subproblems (and, thus, the constraints)
that are automatically derived from
a given model for robust optimization.
Users may specify separation priorities through:

- (Recommended) :class:`~pyomo.core.base.suffix.Suffix` components
  with local name ``pyros_separation_priority``,
  declared on the model or any of its sub-blocks.
  Each entry of every such
  :class:`~pyomo.core.base.suffix.Suffix`
  should map a
  :class:`~pyomo.core.base.var.Var`
  or :class:`~pyomo.core.base.constraint.Constraint`
  component to a value that specifies the separation
  priority of all constraints derived from that component
- The optional argument ``separation_priority_order``
  to the PyROS :py:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve`
  method. The argument should be castable to a :py:obj:`dict`,
  of which each entry maps the full name of a
  :class:`~pyomo.core.base.var.Var`
  or :class:`~pyomo.core.base.constraint.Constraint`
  component to a value that specifies the
  separation priority of all constraints
  derived from that component

Specification via :class:`~pyomo.core.base.suffix.Suffix` components
takes precedence over specification via the solver argument
``separation_priority_order``.
Moreover, the precedence ordering among
:class:`~pyomo.core.base.suffix.Suffix`
components is handled by the Pyomo
:class:`~pyomo.core.base.suffix.SuffixFinder` utility.

A separation priority can be either
a (real) number (i.e., of type :py:class:`int`, :py:class:`float`, etc.)
or :py:obj:`None`.
A higher number indicates a higher priority.
The default priority for all constraints is 0.
Therefore a constraint can be prioritized [or deprioritized]
over the default by mapping the constraint to a positive [or negative] number.
In practice, critical or dominant constraints are often
prioritized over algorithmic or implied constraints.

Constraints that have been assigned a priority of :py:obj:`None`
are enforced subject to only the nominal uncertain parameter realization
provided by the user. Therefore, these constraints are not imposed robustly
and, in particular, are excluded from the separation problems.


.. _pyros_solver_outputs:

Overview of Outputs
===================

.. _pyros_output_results_object:

Results Object
--------------
The :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method returns
an :class:`~pyomo.contrib.pyros.solve_data.ROSolveResults` object.

When the PyROS :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
has successfully solved a given robust optimization problem,
the
:attr:`~pyomo.contrib.pyros.solve_data.ROSolveResults.pyros_termination_condition`
attribute of the returned
:attr:`~pyomo.contrib.pyros.solve_data.ROSolveResults`
object is set to
:attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_optimal`
if and only if:

1. Master problems are solved to global optimality
   (by passing ``solve_master_globally=True``)
2. A worst-case objective focus is chosen
   (by setting ``objective_focus``
   to :attr:`~pyomo.contrib.pyros.util.ObjectiveType.worst_case`)

Otherwise, the termination condition is set to
:attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_feasible`.

The
:attr:`~pyomo.contrib.pyros.solve_data.ROSolveResults.final_objective_value`
attribute of the results object depends on
the value of the optional ``objective_focus`` argument to the
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method:

* If ``objective_focus`` is set to
  :attr:`~pyomo.contrib.pyros.util.ObjectiveType.nominal`,
  then the objective is evaluated subject to
  the nominal uncertain parameter realization
* If ``objective_focus`` is set to
  :attr:`~pyomo.contrib.pyros.util.ObjectiveType.worst_case`,
  then the objective is evaluated subject to
  the uncertain parameter realization that induces the worst-case
  objective value

The second-stage variable and state variable values in the
:ref:`solution loaded to the model <pyros_output_final_solution>`
are evaluated similarly.

.. _pyros_output_final_solution:

Final Solution
--------------
PyROS automatically loads the final solution found to the model
(i.e., updates the values of the variables of the deterministic model)
if and only if:

1. The argument ``load_solution=True`` has been passed to PyROS
   (occurs by default)
2. The
   :attr:`~pyomo.contrib.pyros.solve_data.ROSolveResults.pyros_termination_condition`
   attribute of the returned
   :attr:`~pyomo.contrib.pyros.solve_data.ROSolveResults` object
   is either
   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_optimal`
   or 
   :attr:`~pyomo.contrib.pyros.util.pyrosTerminationCondition.robust_feasible`

Otherwise, the solution is lost.

If a solution is loaded to the model,
then,
as mentioned in our discussion of the
:ref:`results object <pyros_output_results_object>`,
the second-stage variables and state variables
of the model are updated according to
the value of the optional ``objective_focus`` argument to
the  :meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method.
The uncertain parameter objects are left unchanged.


Solver Output Log
-----------------
When the PyROS
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method
is invoked to solve an RO problem,
the progress and final result are reported through a highly
configurable logging system.
See the :ref:`Solver Output Log documentation <pyros_solver_log>`
for more information.
