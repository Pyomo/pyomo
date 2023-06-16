Special Ordered Sets (SOS)
==========================

Pyomo allows users to declare special ordered sets (SOS) within their problems.
These are sets of variables among which only a certain number of variables can
be non-zero and those that are must be adjacent according to a given order. 

Special ordered sets of types 1 (SOS1) and 2 (SOS2) are the most common but the 
concept can be generalised: a SOS of type N cannot have more than N of its 
members taking non-zero values and those that do must be adjacent in the set. 
These can be useful for modelling and computational performance purposes.

By explicitly declaring these, users can keep their formulations and respective 
solving times shorter than they would otherwise, since the logical constraints
that enforce the SOS do not need to be implemented within the model and are 
instead (ideally) handled algorithmically by the solver.

Special ordered sets can be declared one by one or indexed via other sets.

Non-indexed Special Ordered Sets
--------------------------------

A single SOS of type **n** involving all members of a pyomo Var component can 
be declared in one line:

.. doctest::

    >>> # the set that indexes the variables
    >>> model.A = pyo.Set() 
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the sos constraint
    >>> model.mysos = pyo.SOSConstraint(var=model.x, sos=n)

In the example above, the weight of each variable is determined automatically
based on their position/order in the pyomo Var component (``model.x``). 

Alternatively, the weights can be specified through a pyomo Param component
(``model.mysosweights``) indexed by the set also indexing the variables 
(``model.A``):

.. doctest::

    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the weights for each variable used in the sos constraints
    >>> model.mysosweights = pyo.Param(model.A)
    >>> # the sos constraint
    >>> model.mysos = pyo.SOSConstraint(
    ...     var=model.x,
    ...     sos=n,
    ...     weights=model.mysosweights
    ...     )

Indexed Special Ordered Sets
----------------------------

Multiple SOS of type **n** involving members of a pyomo Var component 
(``model.x``) can be created using two additional sets (``model.A`` and 
``model.mysosvarindexset``):

.. doctest::

    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the set indexing the sos constraints
    >>> model.B = pyo.Set()
    >>> # the sets containing the variable indexes for each constraint
    >>> model.mysosvarindexset = pyo.Set(model.B)
    >>> # the sos constraints
    >>> model.mysos = pyo.SOSConstraint(
    ...     model.B,
    ...     var=model.y,
    ...     sos=n,
    ...     index=model.mysosvarindexset
    ...     )

In the example above, the weights are determined automatically from the 
position of the variables. Alternatively, they can be specified through a pyomo
Param component (``model.mysosweights``) and an additional set (``model.C``):

.. doctest::

    >>> # the set that indexes the variables
    >>> model.A = pyo.Set()
    >>> # the variables under consideration
    >>> model.x = pyo.Var(model.A)
    >>> # the set indexing the sos constraints
    >>> model.B = pyo.Set()
    >>> # the sets containing the variable indexes for each constraint
    >>> model.mysosvarindexset = pyo.Set(model.B)
    >>> # the set that indexes the variables used in the sos constraints
    >>> model.C = pyo.Set(within=model.A)
    >>> # the weights for each variable used in the sos constraints
    >>> model.mysosweights = pyo.Param(model.C)
    >>> # the sos constraints
    >>> model.mysos = pyo.SOSConstraint(
    ...     model.B,
    ...     var=model.y,
    ...     sos=n,
    ...     index=model.mysosvarindexset,
    ...     weights=model.mysosweights,
    ...     )

Compatible solvers
------------------

Not all LP/MILP solvers are compatible with SOS declarations and Pyomo might
not be ready to interact with all those that are. The following is a list of 
solvers known to be compatible with special ordered sets:

- CBC
- SCIP
- Gurobi
- CPLEX

Please note that declaring an SOS is no guarantee that a solver will use it as
such in the end. Some solvers, namely Gurobi and CPLEX, might reformulate 
problems with explicit SOS declarations, if they perceive that to be useful.

Example
-------

.. doctest::

    >>> import pyomo.environ as pyo
    >>> sos_type = 1 # or 2
    >>> model = pyo.ConcreteModel()
    >>> model.x = pyo.Var([1], domain=pyo.NonNegativeReals, bounds=(0,40))
    >>> model.A = pyo.Set(initialize=[1,2,4,6])
    >>> model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals)
    >>> model.OBJ = pyo.Objective(
    ...     expr=(1*model.x[1]+
    ...           2*model.y[1]+
    ...           3*model.y[2]+
    ...           -0.1*model.y[4]+
    ...           0.5*model.y[6])
    ...     )
    >>> # add upper bounds
    >>> model.ConstraintY1_ub = pyo.Constraint(expr = model.y[1] <= 2)
    >>> model.ConstraintY2_ub = pyo.Constraint(expr = model.y[2] <= 2)
    >>> model.ConstraintY4_ub = pyo.Constraint(expr = model.y[4] <= 2)
    >>> model.ConstraintY6_ub = pyo.Constraint(expr = model.y[6] <= 2)
    >>> # force some variables to be non-zero
    >>> model.ConstraintYmin = pyo.Constraint(
    ...     expr = (model.x[1]+
    ...             model.y[1]+
    ...             model.y[2]+
    ...             model.y[6] >= 0.25
    ...             )
    ...     )
    >>> # declare sos members
    >>> model.mysos = pyo.SOSConstraint(
    ...     var=model.y, 
    ...     sos=sos_type
    ...     )
