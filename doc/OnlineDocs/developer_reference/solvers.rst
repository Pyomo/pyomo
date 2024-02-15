Future Solver Interface Changes
===============================

Pyomo offers interfaces into multiple solvers, both commercial and open source.
To support better capabilities for solver interfaces, the Pyomo team is actively
redesigning the existing interfaces to make them more maintainable and intuitive
for use. Redesigned interfaces can be found in ``pyomo.contrib.solver``.

.. currentmodule:: pyomo.contrib.solver


New Interface Usage
-------------------

The new interfaces have two modes: backwards compatible and future capability.
To use the backwards compatible version, simply use the ``SolverFactory``
as usual and replace the solver name with the new version. Currently, the new
versions available are:

.. list-table:: Available Redesigned Solvers
   :widths: 25 25
   :header-rows: 1

   * - Solver
     - ``SolverFactory`` Name
   * - ipopt
     - ``ipopt_v2``
   * - GUROBI
     - ``gurobi_v2``

Backwards Compatible Mode
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pyomo.environ as pyo
   from pyomo.contrib.solver.util import assert_optimal_termination

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   status = pyo.SolverFactory('ipopt_v2').solve(model)
   assert_optimal_termination(status)
   model.pprint()

Future Capability Mode
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pyomo.environ as pyo
   from pyomo.contrib.solver.util import assert_optimal_termination
   from pyomo.contrib.solver.ipopt import ipopt

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   opt = ipopt()
   status = opt.solve(model)
   assert_optimal_termination(status)
   # Displays important results information; only available in future capability mode
   status.display()
   model.pprint()


Interface Implementation
------------------------

All new interfaces should be built upon one of two classes (currently):
:class:`SolverBase<pyomo.contrib.solver.base.SolverBase>` or
:class:`PersistentSolverBase<pyomo.contrib.solver.base.PersistentSolverBase>`.

All solvers should have the following:

.. autoclass:: pyomo.contrib.solver.base.SolverBase
   :members:

Persistent solvers include additional members as well as other configuration options:

.. autoclass:: pyomo.contrib.solver.base.PersistentSolverBase
   :show-inheritance:
   :members:

Results
-------

Every solver, at the end of a
:meth:`solve<pyomo.contrib.solver.base.SolverBase.solve>` call, will
return a :class:`Results<pyomo.contrib.solver.results.Results>`
object.  This object is a :py:class:`pyomo.common.config.ConfigDict`,
which can be manipulated similar to a standard ``dict`` in Python.

.. autoclass:: pyomo.contrib.solver.results.Results
   :show-inheritance:
   :members:
   :undoc-members:


Termination Conditions
^^^^^^^^^^^^^^^^^^^^^^

Pyomo offers a standard set of termination conditions to map to solver
returns. The intent of
:class:`TerminationCondition<pyomo.contrib.solver.results.TerminationCondition>`
is to notify the user of why the solver exited. The user is expected
to inspect the :class:`Results<pyomo.contrib.solver.results.Results>`
object or any returned solver messages or logs for more information.

.. autoclass:: pyomo.contrib.solver.results.TerminationCondition
   :show-inheritance:


Solution Status
^^^^^^^^^^^^^^^

Pyomo offers a standard set of solution statuses to map to solver
output. The intent of
:class:`SolutionStatus<pyomo.contrib.solver.results.SolutionStatus>`
is to notify the user of what the solver returned at a high level. The
user is expected to inspect the
:class:`Results<pyomo.contrib.solver.results.Results>` object or any
returned solver messages or logs for more information.

.. autoclass:: pyomo.contrib.solver.results.SolutionStatus
   :show-inheritance:


Solution
--------

Solutions can be loaded back into a model using a ``SolutionLoader``. A specific
loader should be written for each unique case. Several have already been
implemented. For example, for ``ipopt``:

.. autoclass:: pyomo.contrib.solver.ipopt.ipoptSolutionLoader
   :show-inheritance:
   :members:
   :inherited-members:
