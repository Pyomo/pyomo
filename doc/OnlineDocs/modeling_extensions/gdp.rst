Generalized Disjunctive Programming
===================================

The Pyomo.GDP modeling extension gives users the ability to intuitively model
Generalized Disjunctive Programming (GDP) problems--optimization models that
include logical disjunctions (OR-statements). These OR statements arise from
problems that have discrete decisions, for example, the choice of facility
location may be modeled as: :math:`[here] \vee [there] \vee [elsewhere]`.

Modeling
--------

In lieu of a more comprehensive explanation to be written later, we illustrate
with some example models how the GDP extension may be used. The process
synthesis model below is a classic literature example concerning selection
among eight potential process units. Disjunctions govern the selection or
absence of each unit. Disjuncts are the individual terms of each disjunction.
Therefore, disjunctions are composed of an OR relation between various
disjuncts.

In Pyomo, Disjunct is a subclass of Block. You can see below that a Disjunct
may be declared and then subsequently populated with Constraint objects. To
complete the GDP logic, a list of disjuncts is given as the argument to a
Disjunction object, which is the component that actually enforces the logical
OR.

You may think of a Disjunct as a grouping of constraints. Declaring a disjunct
by itself does nothing, and they are ignored by most solvers. Once a
Disjunction is declared, at least one of the constraint groups (disjuncts) must
be enforced. If the disjunction is an exclusive-OR (XOR), then exactly one of
the disjuncts will be enforced.

.. literalinclude:: ../../../examples/gdp/eight_process/eight_proc_model.py

Solving GDP models
------------------

Two approaches are possible for solving GDP models. The traditional approach is
to reformulate the problem to algebraic form (MILP/MINLP). Pyomo provides
callable transformations that can be invoked to perform these reformulations.
Pyomo also provides an alterative approach to solving GDP models via direct
logic-based solution algorithms. These solvers operate directly on the
disjunctive structure of the problem.

GDP to Algebraic transformations
********************************

.. code-block::
  :caption: Big-M Reformulation

  >>> TransformationFactory('gdp.bigm').apply_to(model)

.. code-block::
  :caption: Hull Reformulation

  >>> TransformationFactory('gdp.chull').apply_to(model)

The resulting model object will then be an MILP/MINLP and may be sent to a
traditional solver:

.. code-block::
  :caption: After reformulation

  # tee=True displays BARON output
  >>> SolverFactory('gams:baron').solve(model, tee=True)

Direct GDP solvers
******************

Pyomo provides the GDPopt solver for direct solution of GDP models. Usage of
GDPopt is similar to that of other solvers. The sub-solvers for MILP and NLP
may also be specified:

.. code-block::
  :caption: GDPopt usage

  # model = GDP-model, without algebraic reformulation
  # tee=True displays GDPopt output
  >>> SolverFactory('gdpopt').solve(model, tee=True, mip=gurobi, nlp=ipopt)
