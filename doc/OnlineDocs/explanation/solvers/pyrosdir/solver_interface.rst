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

.. code::

  >>> import pyomo.environ as pyo
  >>> import pyomo.contrib.pyros as pyros  # register the PyROS solver
  >>> pyros_solver = pyo.SolverFactory("pyros")


Overview of Inputs
==================
Deterministic Model
-------------------
PyROS is designed to operate on a single-objective deterministic model,
from which the robust optimization counterpart is automatically inferred.
All variables of the model should be continuous, as
mixed-integer problems are not supported.

First-Stage and Second-Stage Variables
--------------------------------------
A model may have either first-stage variables, second-stage variables,
or both.
Any variable of the model that is excluded from the lists
of first-stage and second-stage variables
is automatically considered to be a state variable.
PyROS assumes that the state variables are
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


.. code::

   import pyomo.environ as pyo
   pyo.Param.DefaultMutable = True

All :class:`~pyomo.core.base.param.Param` objects declared
after the preceding code statements will be made mutable by default.


Uncertainty Set
---------------
See the :ref:`pyros_uncertainty_sets` section.

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

Optional Arguments
------------------
The optional arguments are enumerated in the documentation of the
:meth:`~pyomo.contrib.pyros.pyros.PyROS.solve` method.

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
  then those variables are evaluated at
  the nominal uncertain parameter realization
* If ``objective_focus`` is set to
  :attr:`~pyomo.contrib.pyros.util.ObjectiveType.worst_case`,
  then those variables are evaluated at
  the uncertain parameter realization that induces the worst-case
  objective function value

The second-stage variable and state variable values in the
:ref:`solution loaded to the model <pyros_output_final_solution>`
are evaluated similarly.

.. _pyros_output_final_solution:

Final Solution
--------------
PyROS automatically loads the final solution found to the model
(i.e., updates the values of the variables of the determinstic model)
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


Solver Log Output
-----------------
See the :ref:`pyros_solver_log` section for more information on the
PyROS solver log output.
