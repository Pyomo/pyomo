Future Solver Interface Changes
===============================

.. note::

   The new solver interfaces are still under active development.  They
   are included in the releases as development previews.  Please be
   aware that APIs and functionality may change with no notice.

   We welcome any feedback and ideas as we develop this capability.
   Please post feedback on
   `Issue 1030 <https://github.com/Pyomo/pyomo/issues/1030>`_.

Pyomo offers interfaces into multiple solvers, both commercial and open
source.  To support better capabilities for solver interfaces, the Pyomo
team is actively redesigning the existing interfaces to make them more
maintainable and intuitive for use. A preview of the redesigned
interfaces can be found in ``pyomo.contrib.solver``.

.. currentmodule:: pyomo.contrib.solver


New Interface Usage
-------------------

The new interfaces are not completely backwards compatible with the
existing Pyomo solver interfaces.  However, to aid in testing and
evaluation, we are distributing versions of the new solver interfaces
that are compatible with the existing ("legacy") solver interface.
These "legacy" interfaces are registered with the current
``SolverFactory`` using slightly different names (to avoid conflicts
with existing interfaces).

.. |br| raw:: html

   <br />

.. list-table:: Available Redesigned Solvers and Names Registered
                in the SolverFactories
   :header-rows: 1

   * - Solver
     - Name registered in the |br| ``pyomo.contrib.solver.factory.SolverFactory``
     - Name registered in the |br| ``pyomo.opt.base.solvers.LegacySolverFactory``
   * - Ipopt
     - ``ipopt``
     - ``ipopt_v2``
   * - Gurobi
     - ``gurobi``
     - ``gurobi_v2``

Using the new interfaces through the legacy interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use the new interface as exposed through the existing (legacy)
solver factory and solver interface wrapper.  This provides an API that
is compatible with the existing (legacy) Pyomo solver interface and can
be used with other Pyomo tools / capabilities.

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

Using the new interfaces directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use the new interface by importing it directly:

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
   # Displays important results information; only available through the new interfaces
   status.display()
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   solution_loader: ...
   ...
   3 Declarations: x y obj

Using the new interfaces through the "new" SolverFactory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use the new interface by retrieving it from the new ``SolverFactory``:

.. testcode::
   :skipif: not ipopt_available

   # Import through new SolverFactory
   import pyomo.environ as pyo
   from pyomo.contrib.solver.util import assert_optimal_termination
   from pyomo.contrib.solver.factory import SolverFactory

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   opt = SolverFactory('ipopt')
   status = opt.solve(model)
   assert_optimal_termination(status)
   # Displays important results information; only available through the new interfaces
   status.display()
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   solution_loader: ...
   ...
   3 Declarations: x y obj

Switching all of Pyomo to use the new interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also provide a mechanism to get a "preview" of the future where we
replace the existing (legacy) SolverFactory and utilities with the new
(development) version (see :doc:`future`):

.. testcode::
   :skipif: not ipopt_available

   # Change default SolverFactory version
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
   # Displays important results information; only available through the new interfaces
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

The new interface allows access to new capabilities in the various
problem writers, including the linear presolve and scaling options
recently incorporated into the redesigned NL writer.  For example, you
can control the NL writer in the new ``ipopt`` interface through the
solver's ``writer_config`` configuration option:

.. autoclass:: pyomo.contrib.solver.ipopt.Ipopt
   :members: solve

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
