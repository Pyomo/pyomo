Infeasibility Diagnostics
!!!!!!!!!!!!!!!!!!!!!!!!!

There are two closely related tools for infeasibility diagnosis:

  - :ref:`iis`
  - :ref:`mis`

The first simply provides a conduit for solvers that compute an
infeasible irreducible system (e.g., Cplex, Gurobi, or Xpress).  The
second provides similar functionality, but uses the ``mis`` package
contributed to Pyomo.


.. _iis:

Infeasible Irreducible System (IIS) Tool
========================================

.. automodule:: pyomo.contrib.iis.iis
   :noindex:

.. autofunction:: pyomo.contrib.iis.write_iis
   :noindex:

.. _mis:

Minimal Intractable System finder (MIS) Tool
============================================

The file ``mis.py`` finds sets of actions that each, independently,
would result in feasibility.  The zero-tolerance is whatever the
solver uses, so users may want to post-process output if it is going
to be used for analysis. It also computes a minimal intractable system
(which is not guaranteed to be unique).  It was written by Ben Knueven
as part of the watertap project (https://github.com/watertap-org/watertap)
and is therefore governed by a license shown
at the top of ``mis.py``.

The algorithms come from John Chinneck's slides, see: http://www.sce.carleton.ca/faculty/chinneck/docs/CPAIOR07InfeasibilityTutorial.pdf

Solver
------

At the time of this writing, you need to use IPopt even for LPs.

Quick Start
-----------

The file ``trivial_mis.py`` is a tiny example listed at the bottom of
this help file, which references a Pyomo model with the Python variable
`m` and has these lines:

.. code-block:: python

   from pyomo.contrib.mis import compute_infeasibility_explanation
   ipopt = pyo.SolverFactory("ipopt")
   compute_infeasibility_explanation(m, solver=ipopt)

.. Note::
   This is done instead of solving the problem.
   
.. Note::
   IDAES users can pass ``get_solver()`` imported from ``ideas.core.solvers``
   as the solver.

Interpreting the Output
-----------------------

Assuming the dependencies are installed, running ``trivial_mis.py``
(shown below) will
produce a lot of warnings from IPopt and then meaningful output (using a logger).

Repair Options
^^^^^^^^^^^^^^

This output for the trivial example shows three independent ways that the model could be rendered feasible:


.. code-block:: text
   
   Model Trivial Quad may be infeasible. A feasible solution was found with only the following variable bounds relaxed:
	ub of var x[1] by 4.464126126706818e-05
	lb of var x[2] by 0.9999553410114216
   Another feasible solution was found with only the following variable bounds relaxed:
	lb of var x[1] by 0.7071067726864677
	ub of var x[2] by 0.41421355687130673
	ub of var y by 0.7071067651855212
   Another feasible solution was found with only the following inequality constraints, equality constraints, and/or variable bounds relaxed:
	constraint: c by 0.9999999861866736


Minimal Intractable System (MIS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This output shows a minimal intractable system:


.. code-block:: text

   Computed Minimal Intractable System (MIS)!
   Constraints / bounds in MIS:
	lb of var x[2]
	lb of var x[1]
	constraint: c
   
Constraints / bounds in guards for stability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This part of the report is for nonlinear programs (NLPs).

When we’re trying to reduce the constraint set, for an NLP there may be constraints that when missing cause the solver 
to fail in some catastrophic fashion. In this implementation this is interpreted as failing to get a `results` 
object back from the call to `solve`. In these cases we keep the constraint in the problem but it’s in the 
set of “guard” constraints – we can’t really be sure they’re a source of infeasibility or not, 
just that “bad things” happen when they’re not included.

Perhaps ideally we would put a constraint in the “guard” set if IPopt failed to converge, and only put it in the 
MIS if IPopt converged to a point of local infeasibility. However, right now the code generally makes the 
assumption that if IPopt fails to converge the subproblem is infeasible, though obviously that is far from the truth. 
Hence for difficult NLPs even the “Phase 1” may “fail” – in that when finished the subproblem containing just the 
constraints in the elastic filter may be feasible -- because IPopt failed to converge and we assumed that meant the 
subproblem was not feasible.

Dealing with NLPs is far from clean, but that doesn’t mean the tool can’t return useful results even when its assumptions are not satisfied.

trivial_mis.py
--------------

.. code-block:: python

   import pyomo.environ as pyo
   m = pyo.ConcreteModel("Trivial Quad")
   m.x = pyo.Var([1,2], bounds=(0,1))
   m.y = pyo.Var(bounds=(0, 1))
   m.c = pyo.Constraint(expr=m.x[1] * m.x[2] == -1)
   m.d = pyo.Constraint(expr=m.x[1] + m.y >= 1)

   from pyomo.contrib.mis import compute_infeasibility_explanation
   ipopt = pyo.SolverFactory("ipopt")
   compute_infeasibility_explanation(m, solver=ipopt)
