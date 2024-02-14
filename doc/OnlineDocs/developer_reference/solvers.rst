Solver Interfaces
=================

Pyomo offers interfaces into multiple solvers, both commercial and open source.

.. currentmodule:: pyomo.contrib.solver


Interface Implementation
------------------------

All new interfaces should be built upon one of two classes (currently):
``pyomo.contrib.solver.base.SolverBase`` or ``pyomo.contrib.solver.base.PersistentSolverBase``.

All solvers should have the following:

.. autoclass:: pyomo.contrib.solver.base.SolverBase
   :members:

Persistent solvers should also include:

.. autoclass:: pyomo.contrib.solver.base.PersistentSolverBase
   :show-inheritance:
   :members:

Results
-------

Every solver, at the end of a ``solve`` call, will return a ``Results`` object.
This object is a :py:class:`pyomo.common.config.ConfigDict`, which can be manipulated similar
to a standard ``dict`` in Python.

.. autoclass:: pyomo.contrib.solver.results.Results
   :show-inheritance:
   :members:
   :undoc-members:


Termination Conditions
^^^^^^^^^^^^^^^^^^^^^^

Pyomo offers a standard set of termination conditions to map to solver
returns. The intent of ``TerminationCondition`` is to notify the user of why
the solver exited. The user is expected to inspect the ``Results`` object or any
returned solver messages or logs for more information.



.. autoclass:: pyomo.contrib.solver.results.TerminationCondition
   :show-inheritance:
   :noindex:


Solution Status
^^^^^^^^^^^^^^^

Pyomo offers a standard set of solution statuses to map to solver output. The
intent of ``SolutionStatus`` is to notify the user of what the solver returned
at a high level. The user is expected to inspect the ``Results`` object or any
returned solver messages or logs for more information.

.. autoclass:: pyomo.contrib.solver.results.SolutionStatus
   :show-inheritance:
   :noindex:


Solution
--------

Solutions can be loaded back into a model using a ``SolutionLoader``. A specific
loader should be written for each unique case. Several have already been
implemented. For example, for ``ipopt``:

.. autoclass:: pyomo.contrib.solver.solution.SolSolutionLoader
   :show-inheritance:
   :members:
