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
The future capability mode can be accessed directly or by switching the default
``SolverFactory`` version (see :doc:`future`). Currently, the new versions
available are:

.. list-table:: Available Redesigned Solvers
   :widths: 25 25 25
   :header-rows: 1

   * - Solver
     - ``SolverFactory`` (v1) Name
     - ``SolverFactory`` (v3) Name
   * - ipopt
     - ``ipopt_v2``
     - ``ipopt``
   * - Gurobi
     - ``gurobi_v2``
     - ``gurobi``

Backwards Compatible Mode
^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode::
   :skipif: not ipopt_available

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

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   2 Var Declarations
   ...
   3 Declarations: x y obj

Future Capability Mode
^^^^^^^^^^^^^^^^^^^^^^

There are multiple ways to utilize the future capability mode: direct import
or changed ``SolverFactory`` version.

.. testcode::
   :skipif: not ipopt_available

    # Direct import
   import pyomo.environ as pyo
   from pyomo.contrib.solver.util import assert_optimal_termination
   from pyomo.contrib.solver.ipopt import Ipopt

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   opt = Ipopt()
   status = opt.solve(model)
   assert_optimal_termination(status)
   # Displays important results information; only available in future capability mode
   status.display()
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   solution_loader: ...
   ...
   3 Declarations: x y obj

Changing the ``SolverFactory`` version:

.. testcode::
   :skipif: not ipopt_available

    # Change SolverFactory version
   import pyomo.environ as pyo
   from pyomo.contrib.solver.util import assert_optimal_termination
   from pyomo.__future__ import solver_factory_v3

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   status = pyo.SolverFactory('ipopt').solve(model)
   assert_optimal_termination(status)
   # Displays important results information; only available in future capability mode
   status.display()
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   solution_loader: ...
   ...
   3 Declarations: x y obj

.. testcode::
   :skipif: not ipopt_available
   :hide:

   from pyomo.__future__ import solver_factory_v1

Linear Presolve and Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The new interface will allow for direct manipulation of linear presolve and scaling
options for certain solvers. Currently, these options are only available for
``ipopt``.

.. autoclass:: pyomo.contrib.solver.ipopt.Ipopt
   :members: solve

The ``writer_config`` configuration option can be used to manipulate presolve
and scaling options:

.. testcode::

   from pyomo.contrib.solver.ipopt import Ipopt
   opt = Ipopt()
   opt.config.writer_config.display()

.. testoutput::

   show_section_timing: false
   skip_trivial_constraints: true
   file_determinism: FileDeterminism.ORDERED
   symbolic_solver_labels: false
   scale_model: true
   export_nonlinear_variables: None
   row_order: None
   column_order: None
   export_defined_variables: true
   linear_presolve: true

Note that, by default, both ``linear_presolve`` and ``scale_model`` are enabled.
Users can manipulate ``linear_presolve`` and ``scale_model`` to their preferred
states by changing their values.

.. code-block:: python

   >>> opt.config.writer_config.linear_presolve = False


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

.. autoclass:: pyomo.contrib.solver.ipopt.IpoptSolutionLoader
   :show-inheritance:
   :members:
   :inherited-members:
