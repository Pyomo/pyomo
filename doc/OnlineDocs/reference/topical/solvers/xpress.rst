Xpress (New Interface)
======================

.. currentmodule:: pyomo.contrib.solver.solvers.xpress

Pyomo provides two solver interfaces to the FICO Xpress solver:
:class:`XpressDirect` for one-shot solves, and :class:`XpressPersistent`
for workflows that solve a model repeatedly with small modifications
between solves.

Both connectors support the complete range of problem classes that Xpress
handles: LP, MIP, QP, MIQP, NLP, MINLP, second-order cone programs,
and SOS Type 1 and 2 constraints.

Expression Walker
-----------------

:class:`XpressDirect` uses a custom expression walker that translates the
full Pyomo expression tree (linear, quadratic, or nonlinear) directly
into an equivalent Xpress expression object, avoiding further intermediate 
Python transformations, and handing off to the Xpress C library as directly
as possible. Quadratic terms arising from Cartesian-product expansions are
expanded on the C side. The result is a lean, single-path translation
with no additional overhead for more complex expression types.

:class:`XpressPersistent` takes a slightly different approach.
Pyomo's ``generate_standard_repn`` runs first: it decomposes each
constraint into its linear and quadratic parts and, crucially, provides
symbolic (non-evaluated) coefficients that are used to register the
mutable-parameter update helpers driving the targeted ``chgMCoef`` /
``chgRHS`` / ``chgQRowCoeff`` calls between solves. If a nonlinear
subexpression remains after that decomposition, the same walker handles
it, producing an Xpress nonlinear expression. 

XpressDirect
------------

:class:`XpressDirect` builds a fresh Xpress problem from the Pyomo model
on every call to :meth:`~XpressDirect.solve`. Use it for one-shot solves
or exploratory modeling.

.. code-block:: python

   from pyomo.contrib.solver.solvers.xpress import XpressDirect
   import pyomo.environ as pyo

   m = pyo.ConcreteModel()
   m.x = pyo.Var(bounds=(0, 10))
   m.c = pyo.Constraint(expr=m.x >= 3)
   m.obj = pyo.Objective(expr=m.x)

   res = XpressDirect().solve(m)

XpressPersistent
----------------

:class:`XpressPersistent` keeps the Xpress problem in memory between
solves and uses Pyomo's model-change notification framework to apply
only the minimal set of solver API calls required to reflect each change.

Mutable :class:`~pyomo.environ.Param` components are tracked
automatically. Updating a parameter value before the next
:meth:`~XpressPersistent.solve` call triggers targeted coefficient or
bound updates (``chgMCoef``, ``chgRHS``, ``chgQRowCoeff``) rather than a
full model rebuild.

.. code-block:: python

   from pyomo.contrib.solver.solvers.xpress import XpressPersistent
   import pyomo.environ as pyo

   m = pyo.ConcreteModel()
   m.cost = pyo.Param(mutable=True, initialize=2.0)
   m.x = pyo.Var(bounds=(0, 10))
   m.c = pyo.Constraint(expr=m.x >= 3)
   m.obj = pyo.Objective(expr=m.cost * m.x)

   opt = XpressPersistent()
   opt.solve(m)           # full build
   m.cost.set_value(5.0)
   opt.solve(m)           # incremental: only the objective coefficient is updated

Incremental operations
^^^^^^^^^^^^^^^^^^^^^^

Between solves the persistent connector supports:

- **LP/QP coefficient and bound updates** without row removal, using
  the Xpress ``chgMCoef`` / ``chgRHS`` / ``chgQRowCoeff`` API.
- **NLP constraint updates** via row removal and re-insertion (required
  when the nonlinear structure changes).
- **Variable fixing and unfixing** through bound updates only. No
  constraints are removed or rebuilt as a result of fixing; Xpress
  folds fixed variables natively during the solve. Fixing all integer
  variables in a MINLP therefore reduces to a sequence of bound calls,
  after which Xpress can treat the problem as continuous without any
  structural modification to the Pyomo model.
- **Structural modifications**: add and remove constraints, variables,
  SOS constraints, and sub-blocks.

Configuration
-------------

Both connectors accept a common set of configuration options passed as
keyword arguments to :meth:`~XpressDirect.solve`. Options labelled
*framework* are defined in the Pyomo solver framework base class and are
expected to be supported by every compliant connector. Options labelled
*connector* are specific to this Xpress implementation.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Option
     - Scope
     - Description
   * - ``time_limit``
     - framework
     - Wall-clock time limit in seconds.
   * - ``threads``
     - framework
     - Number of solver threads.
   * - ``rel_gap``
     - framework
     - Relative MIP optimality gap tolerance.
   * - ``abs_gap``
     - framework
     - Absolute MIP optimality gap tolerance.
   * - ``symbolic_solver_labels``
     - framework
     - Use Pyomo component names in the Xpress problem (aids debugging).
   * - ``solver_options``
     - framework
     - Dict of raw solver control names forwarded directly to the solver.
   * - ``warmstart``
     - connector
     - Pass variable values as a MIP start hint (default ``True``).
   * - ``pool_solutions``
     - connector
     - Collect multiple MIP solutions during B&B (0 = disabled). ``N > 0``:
       keep a rolling window of the last ``N`` solutions found.

Any Xpress control name accepted by ``prob.controls.<name>`` can be passed:

.. code-block:: python

   res = opt.solve(m,
       solver_options={
           'outputlog': 0,       # suppress solver output
           'maxnode': 500,       # B&B node limit
           'feastol': 1e-8,      # primal feasibility tolerance
       }
   )

Results
-------

Every :meth:`~XpressDirect.solve` call returns a
:class:`~pyomo.contrib.solver.common.results.Results` object:

.. code-block:: python

   res = opt.solve(m)
   print(res.termination_condition)   # e.g. convergenceCriteriaSatisfied
   print(res.solution_status)         # e.g. optimal
   print(res.incumbent_objective)     # objective value at the best solution

:attr:`~pyomo.contrib.solver.common.results.Results.termination_condition`
reports why the solver stopped;
:attr:`~pyomo.contrib.solver.common.results.Results.solution_status`
reports what was returned. For NLP problems solved via Xpress SLP,
``solution_status`` will be ``feasible`` rather than ``optimal``,
reflecting the local convergence nature of the algorithm.

Solution Pool
-------------

:class:`XpressDirect` and :class:`XpressPersistent` can collect multiple
feasible MIP solutions found during branch-and-bound via the
``pool_solutions`` configuration option. Setting ``pool_solutions=N``
(N > 0) keeps a rolling window of the last ``N`` solutions found: once
the window is full, the oldest entry is evicted on each new solution,
so the pool always contains the N most recently discovered feasible
solutions.

.. code-block:: python

   res = opt.solve(m, pool_solutions=5)
   loader = res.solution_loader
   print(loader.get_number_of_solutions())   # up to 6 (incumbent + pool)

   # Load the incumbent (solution 0) into the model
   loader.solution(0).load_vars()

   # Inspect pool entry 1 without modifying the model permanently
   with loader.solution(1):
       loader.load_vars()
       print(m.x.value)
   # After the with-block the active solution reverts to the incumbent

NLP and Nonlinear Expressions
------------------------------

All standard Pyomo nonlinear operators, trigonometric and hyperbolic
functions, and user-defined Python callback functions
(``pyo.ExternalFunction``) are supported.

``pyo.floor`` and ``pyo.ceil`` are not currently supported and raise
:class:`~pyomo.contrib.solver.common.util.IncompatibleModelError`.
These operations must be reformulated by introducing an auxiliary integer
variable together with two linear inequality constraints that encode the
floor or ceil relationship. Adding an integer variable to a continuous
NLP produces a MINLP.

Testing
-------

The connector ships with a test suite covering LP, MIP, QP, QCP, NLP,
MINLP, SOS, mutable parameter tracking, incremental structural updates,
and the solution pool.

.. note::

   Development of this connector was aided by
   `Claude Code <https://claude.ai/code>`_ (Anthropic).

API Reference
-------------

.. autosummary::
   :toctree: generated/

   XpressDirect
   XpressPersistent
