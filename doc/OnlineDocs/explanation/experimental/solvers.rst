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
     - Name registered in the |br| ``pyomo.contrib.solver.common.factory.SolverFactory``
     - Name registered in the |br| ``pyomo.opt.base.solvers.LegacySolverFactory``
   * - Ipopt
     - ``ipopt``
     - ``ipopt_v2``
   * - Gurobi (persistent)
     - ``gurobi_persistent``
     - ``gurobi_persistent_v2``
   * - Gurobi (direct)
     - ``gurobi_direct``
     - ``gurobi_direct_v2``
   * - HiGHS
     - ``highs``
     - ``highs``
   * - KNITRO
     - ``knitro_direct``
     - ``knitro_direct``

Using the new interfaces through the legacy interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use the new interface as exposed through the existing (legacy)
solver factory and solver interface wrapper.  This provides an API that
is compatible with the existing (legacy) Pyomo solver interface and can
be used with other Pyomo tools / capabilities.

.. testcode::
   :skipif: not ipopt_available

   import pyomo.environ as pyo

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   status = pyo.SolverFactory('ipopt_v2').solve(model)
   pyo.assert_optimal_termination(status)
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   2 Var Declarations
   ...
   3 Declarations: x y obj

In keeping with our commitment to backwards compatibility, both the legacy and
future methods of specifying solver options are supported:

.. testcode::
   :skipif: not ipopt_available

   import pyomo.environ as pyo

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   # Backwards compatible
   status = pyo.SolverFactory('ipopt_v2').solve(model, options={'max_iter' : 6})
   # Forwards compatible
   status = pyo.SolverFactory('ipopt_v2').solve(model, solver_options={'max_iter' : 6})
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
   from pyomo.contrib.solver.solvers.ipopt import Ipopt

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   opt = Ipopt()
   status = opt.solve(model)
   pyo.assert_optimal_termination(status)
   # Displays important results information; only available through the new interfaces
   status.display()
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   termination_condition: ...
   ...
   3 Declarations: x y obj

Using the new interfaces through the "new" SolverFactory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use the new interface by retrieving it from the new ``SolverFactory``:

.. testcode::
   :skipif: not ipopt_available

   # Import through new SolverFactory
   import pyomo.environ as pyo
   from pyomo.contrib.solver.common.factory import SolverFactory

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   opt = SolverFactory('ipopt')
   status = opt.solve(model)
   pyo.assert_optimal_termination(status)
   # Displays important results information; only available through the new interfaces
   status.display()
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   termination_condition: ...
   ...
   3 Declarations: x y obj

Switching all of Pyomo to use the new interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also provide a mechanism to get a "preview" of the future where we
replace the existing (legacy) SolverFactory and utilities with the new
(development) version (see :doc:`/reference/future`):

.. testcode::
   :skipif: not ipopt_available

   # Change default SolverFactory version
   import pyomo.environ as pyo
   from pyomo.__future__ import solver_factory_v3

   model = pyo.ConcreteModel()
   model.x = pyo.Var(initialize=1.5)
   model.y = pyo.Var(initialize=1.5)

   def rosenbrock(model):
       return (1.0 - model.x) ** 2 + 100.0 * (model.y - model.x**2) ** 2

   model.obj = pyo.Objective(rule=rosenbrock, sense=pyo.minimize)

   status = pyo.SolverFactory('ipopt').solve(model)
   pyo.assert_optimal_termination(status)
   # Displays important results information; only available through the new interfaces
   status.display()
   model.pprint()

.. testoutput::
   :skipif: not ipopt_available
   :hide:

   termination_condition: ...
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
solver's ``writer_config`` configuration option (see the
:class:`~pyomo.contrib.solver.solvers.ipopt.Ipopt` interface documentation).

.. testcode::

   from pyomo.contrib.solver.solvers.ipopt import Ipopt
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
:class:`SolverBase<pyomo.contrib.solver.common.base.SolverBase>` or
:class:`PersistentSolverBase<pyomo.contrib.solver.common.base.PersistentSolverBase>`.

All solvers should have the following:

.. autoclass:: pyomo.contrib.solver.common.base.SolverBase
   :noindex:
   :members:

Persistent solvers include additional members as well as other configuration options:

.. autoclass:: pyomo.contrib.solver.common.base.PersistentSolverBase
   :noindex:
   :show-inheritance:
   :members:

Results
-------

Every solver, at the end of a
:meth:`solve<pyomo.contrib.solver.common.base.SolverBase.solve>` call, will
return a :class:`Results<pyomo.contrib.solver.common.results.Results>`
object.  This object is a :py:class:`pyomo.common.config.ConfigDict`,
which can be manipulated similar to a standard ``dict`` in Python.

.. autoclass:: pyomo.contrib.solver.common.results.Results
   :noindex:
   :show-inheritance:
   :members:
   :undoc-members:

The new interface has condensed :py:class:`~pyomo.opt.results.solver.SolverStatus`,
:py:class:`~pyomo.opt.results.solver.TerminationCondition`,
and :py:class:`~pyomo.opt.results.solution.SolutionStatus` into
:py:class:`~pyomo.contrib.solver.common.results.TerminationCondition`
and :py:class:`~pyomo.contrib.solver.common.results.SolutionStatus` to
reduce complexity. As a result, several legacy
:py:class:`~pyomo.opt.results.solver.SolutionStatus` values are
no longer achievable. These are detailed in the table below.

.. list-table:: Mapping from unachievable :py:class:`~pyomo.opt.results.solver.SolutionStatus`
                to future statuses
   :header-rows: 1

   * - Legacy :py:class:`~pyomo.opt.results.solver.SolutionStatus`
     - :py:class:`~pyomo.contrib.solver.common.results.TerminationCondition`
     - :py:class:`~pyomo.contrib.solver.common.results.SolutionStatus`
   * - other
     - unknown
     - noSolution
   * - unsure
     - unknown
     - noSolution
   * - locallyOptimal
     - convergenceCriteriaSatisfied
     - optimal
   * - globallyOptimal
     - convergenceCriteriaSatisfied
     - optimal
   * - bestSoFar
     - convergenceCriteriaSatisfied
     - feasible

Termination Conditions
^^^^^^^^^^^^^^^^^^^^^^

Pyomo offers a standard set of termination conditions to map to solver
returns. The intent of
:class:`TerminationCondition<pyomo.contrib.solver.common.results.TerminationCondition>`
is to notify the user of why the solver exited. The user is expected
to inspect the :class:`Results<pyomo.contrib.solver.common.results.Results>`
object or any returned solver messages or logs for more information.

.. autoclass:: pyomo.contrib.solver.common.results.TerminationCondition
   :noindex:
   :show-inheritance:


Solution Status
^^^^^^^^^^^^^^^

Pyomo offers a standard set of solution statuses to map to solver
output. The intent of
:class:`SolutionStatus<pyomo.contrib.solver.common.results.SolutionStatus>`
is to notify the user of what the solver returned at a high level. The
user is expected to inspect the
:class:`Results<pyomo.contrib.solver.common.results.Results>` object or any
returned solver messages or logs for more information.

.. autoclass:: pyomo.contrib.solver.common.results.SolutionStatus
   :noindex:
   :show-inheritance:


Solution
--------

Solutions can be loaded back into a model using a ``SolutionLoader``. A specific
loader should be written for each unique case. Several have already been
implemented. For example, for ``ipopt``:

.. autoclass:: pyomo.contrib.solver.solvers.ipopt.IpoptSolutionLoader
   :noindex:
   :members:
   :show-inheritance:
   :inherited-members:


Dual Sign Convention
--------------------
For all future solver interfaces, Pyomo adopts the following sign convention. Given the problem

.. math::

   \begin{aligned}
   \min\quad      & f(x) \\
   \text{s.t.}\quad & c_i(x) = 0 \quad \forall i \in \mathcal{E} \\
                    & g_i(x) \le 0 \quad \forall i \in \mathcal{U} \\
                    & h_i(x) \ge 0 \quad \forall i \in \mathcal{L}
   \end{aligned}

We define the Lagrangian as

.. math::

   \begin{aligned}
   L(x, \lambda, \nu, \delta)
     &= f(x)
        - \sum_{i \in \mathcal{E}} \lambda_i\,c_i(x)
        - \sum_{i \in \mathcal{U}} \nu_i\,g_i(x)
        - \sum_{i \in \mathcal{L}} \delta_i\,h_i(x)
   \end{aligned}

Then, the KKT conditions are [NW99]_

.. math::

   \begin{aligned}
   \nabla_x L(x, \lambda, \nu, \delta) &= 0 \\
   c(x)                                &= 0 \\
   g(x)                                &\le 0 \\
   h(x)                                &\ge 0 \\
   \nu                                 &\le 0 \\
   \delta                              &\ge 0 \\
   \nu_i\,g_i(x)                       &= 0 \\
   \delta_i\,h_i(x)                    &= 0
   \end{aligned}

Note that this sign convention is based on the ``(lower, body, upper)``
representation of constraints rather than the expression provided by a
user. Users can specify constraints with variables on both the left- and
right-hand sides of equalities and inequalities. However, the
``(lower, body, upper)`` representation ensures that all variables
appear in the ``body``, matching the form of the problem above.

For maximization problems of the form

.. math::

   \begin{aligned}
   \max\quad      & f(x) \\
   \text{s.t.}\quad & c_i(x) = 0 \quad \forall i \in \mathcal{E} \\
                    & g_i(x) \le 0 \quad \forall i \in \mathcal{U} \\
                    & h_i(x) \ge 0 \quad \forall i \in \mathcal{L}
   \end{aligned}

we define the Lagrangian to be the same as above:

.. math::

   \begin{aligned}
   L(x, \lambda, \nu, \delta)
     &= f(x)
        - \sum_{i \in \mathcal{E}} \lambda_i\,c_i(x)
        - \sum_{i \in \mathcal{U}} \nu_i\,g_i(x)
        - \sum_{i \in \mathcal{L}} \delta_i\,h_i(x)
   \end{aligned}

As a result, the signs of the duals change. The KKT conditions are 

.. math::

   \begin{aligned}
   \nabla_x L(x, \lambda, \nu, \delta) &= 0 \\
   c(x)                                &= 0 \\
   g(x)                                &\le 0 \\
   h(x)                                &\ge 0 \\
   \nu                                 &\ge 0 \\
   \delta                              &\le 0 \\
   \nu_i\,g_i(x)                       &= 0 \\
   \delta_i\,h_i(x)                    &= 0
   \end{aligned}


Pyomo also supports "range constraints" which are inequalities with both upper
and lower bounds, where the bounds are not equal. For example,

.. math::

   -1 \leq x + y \leq 1

These are handled very similarly to variable bounds in terms of dual sign
conventions. For these, at most one "side" of the inequality can be active
at a time. If neither side is active, then the dual will be zero. If the dual
is nonzero, then the dual corresponds to the side of the constraint that is
active. The dual for the other side will be implicitly zero. When accessing
duals, the keys are the constraints. As a result, there is only one key for
a range constraint, even though it is really two constraints. Therefore, the
dual for the inactive side will not be reported explicitly. Again, the sign
convention is based on the ``(lower, body, upper)`` representation of the
constraint. Therefore, the left side of this inequality belongs to
:math:`\mathcal{L}` and the right side belongs to :math:`\mathcal{U}`.
