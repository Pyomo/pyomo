Special Ordered Sets (SOS)
==========================

Pyomo allows users to declare special ordered sets (SOS) within their problems.
These are sets of variables among which only a certain number of variables can
be non-zero, and those that are must be adjacent according to a given order. 

Special ordered sets of types 1 (SOS1) and 2 (SOS2) are the classic ones, but 
the concept can be generalised: a SOS of type N cannot have more than N of its 
members taking non-zero values, and those that do must be adjacent in the set. 
These can be useful for modelling and computational performance purposes.

By explicitly declaring these, users can keep their formulations and respective 
solving times shorter than they would otherwise, since the logical constraints
that enforce the SOS do not need to be implemented within the model and are 
instead (ideally) handled algorithmically by the solver.

Special ordered sets can be declared one by one or indexed via other sets.

Non-indexed Special Ordered Sets
--------------------------------

A single SOS of type **N** involving all members of a pyomo Var component can 
be declared in one line:

.. currentmodule:: pyomo.environ

.. testcode::

   # import pyomo
   import pyomo.environ as pyo
   # declare the model
   model = pyo.AbstractModel()
   # the type of SOS
   N = 1 # or 2, 3, ...
   # the set that indexes the variables
   model.A = pyo.Set() 
   # the variables under consideration
   model.x = pyo.Var(model.A)
   # the sos constraint
   model.mysos = pyo.SOSConstraint(var=model.x, sos=N)

In the example above, the weight of each variable is determined automatically
based on their position/order in the pyomo Var component (``model.x``). 

Alternatively, the weights can be specified through a pyomo Param component
(``model.mysosweights``) indexed by the set also indexing the variables 
(``model.A``):

.. doctest::
   :hide:

   >>> model = pyo.AbstractModel()

.. testcode::

   # the set that indexes the variables
   model.A = pyo.Set()
   # the variables under consideration
   model.x = pyo.Var(model.A)
   # the weights for each variable used in the sos constraints
   model.mysosweights = pyo.Param(model.A)
   # the sos constraint
   model.mysos = pyo.SOSConstraint(
       var=model.x,
       sos=N,
       weights=model.mysosweights
       )

Indexed Special Ordered Sets
----------------------------

Multiple SOS of type **N** involving members of a pyomo Var component 
(``model.x``) can be created using two additional sets (``model.A`` and 
``model.mysosvarindexset``):

.. doctest::
   :hide:

   >>> model = pyo.AbstractModel()

.. testcode::

   # the set that indexes the variables
   model.A = pyo.Set()
   # the variables under consideration
   model.x = pyo.Var(model.A)
   # the set indexing the sos constraints
   model.B = pyo.Set()
   # the sets containing the variable indexes for each constraint
   model.mysosvarindexset = pyo.Set(model.B)
   # the sos constraints
   model.mysos = pyo.SOSConstraint(
       model.B,
       var=model.x,
       sos=N,
       index=model.mysosvarindexset
       )

In the example above, the weights are determined automatically from the 
position of the variables. Alternatively, they can be specified through a pyomo
Param component (``model.mysosweights``) and an additional set (``model.C``):

.. doctest::
   :hide:

   >>> model = pyo.AbstractModel()

.. testcode::

   # the set that indexes the variables
   model.A = pyo.Set()
   # the variables under consideration
   model.x = pyo.Var(model.A)
   # the set indexing the sos constraints
   model.B = pyo.Set()
   # the sets containing the variable indexes for each constraint
   model.mysosvarindexset = pyo.Set(model.B)
   # the set that indexes the variables used in the sos constraints
   model.C = pyo.Set(within=model.A)
   # the weights for each variable used in the sos constraints
   model.mysosweights = pyo.Param(model.C)
   # the sos constraints
   model.mysos = pyo.SOSConstraint(
       model.B,
       var=model.x,
       sos=N,
       index=model.mysosvarindexset,
       weights=model.mysosweights,
       )

Declaring Special Ordered Sets using rules
------------------------------------------

Arguably the best way to declare an SOS is through rules. This option allows 
users to specify the variables and weights through a method provided via the 
``rule`` parameter. If this parameter is used, users must specify a method that
returns one of the following options:

- a list of the variables in the SOS, whose respective weights are then determined based on their position;

- a tuple of two lists, the first for the variables in the SOS and the second for the respective weights;

- or, pyomo.environ.SOSConstraint.Skip, if the SOS is not to be declared.

If one is content on having the weights determined based on the position of the
variables, then the following example using the ``rule`` parameter is sufficient:

.. doctest::
   :hide:

   >>> model = pyo.AbstractModel()

.. testcode::

   # the set that indexes the variables
   model.A = pyo.Set()
   # the variables under consideration
   model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)
   # the rule method creating the constraint
   def rule_mysos(m):
       return [m.x[a] for a in m.x]
   # the sos constraint(s)
   model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=N)
   
   
If the weights must be determined in some other way, then the following example
illustrates how they can be specified for each member of the SOS using the ``rule`` parameter:

.. doctest::
   :hide:

   >>> model = pyo.AbstractModel()

.. testcode::

   # the set that indexes the variables
   model.A = pyo.Set()
   # the variables under consideration
   model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)
   # the rule method creating the constraint
   def rule_mysos(m):
       var_list = [m.x[a] for a in m.x]
       weight_list = [i+1 for i in range(len(var_list))]
       return (var_list, weight_list)
   # the sos constraint(s)
   model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=N)
   
The ``rule`` parameter also allows users to create SOS comprising variables 
from different pyomo Var components, as shown below:

.. doctest::
   :hide:

   >>> model = pyo.AbstractModel()

.. testcode::

   # the set that indexes the x variables
   model.A = pyo.Set()
   # the set that indexes the y variables
   model.B = pyo.Set()
   # the set that indexes the SOS constraints
   model.C = pyo.Set()
   # the x variables, which will be used in the constraints
   model.x = pyo.Var(model.A, domain=pyo.NonNegativeReals)
   # the y variables, which will be used in the constraints
   model.y = pyo.Var(model.B, domain=pyo.NonNegativeReals)
   # the x variable indices for each constraint
   model.mysosindex_x = pyo.Set(model.C)
   # the y variable indices for each constraint
   model.mysosindex_y = pyo.Set(model.C)
   # the weights for the x variable indices
   model.mysosweights_x = pyo.Param(model.A)
   # the weights for the y variable indices
   model.mysosweights_y = pyo.Param(model.B)
   # the rule method with which each constraint c is built
   def rule_mysos(m, c):
       var_list = [m.x[a] for a in m.mysosindex_x[c]]
       var_list.extend([m.y[b] for b in m.mysosindex_y[c]])
       weight_list = [m.mysosweights_x[a] for a in m.mysosindex_x[c]]
       weight_list.extend([m.mysosweights_y[b] for b in m.mysosindex_y[c]])
       return (var_list, weight_list)
   # the sos constraint(s)
   model.mysos = pyo.SOSConstraint(
       model.C,
       rule=rule_mysos,
       sos=N
       )

Compatible solvers
------------------

Not all LP/MILP solvers are compatible with SOS declarations and Pyomo might
not be ready to interact with all those that are. The following is a list of 
solvers known to be compatible with special ordered sets through Pyomo:

- CBC
- SCIP
- Gurobi
- CPLEX

Please note that declaring an SOS is no guarantee that a solver will use it as
such in the end. Some solvers, namely Gurobi and CPLEX, might reformulate 
problems with explicit SOS declarations, if they perceive that to be useful.

Full example with non-indexed SOS constraint
--------------------------------------------

.. doctest::
   :hide:

   >>> model = pyo.AbstractModel()

.. testcode::

   import pyomo.environ as pyo
   from pyomo.opt import check_available_solvers
   from math import isclose
   N = 1
   model = pyo.ConcreteModel()
   model.x = pyo.Var([1], domain=pyo.NonNegativeReals, bounds=(0,40))
   model.A = pyo.Set(initialize=[1,2,4,6])
   model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals, bounds=(0,2))
   model.OBJ = pyo.Objective(
       expr=(1*model.x[1]+
             2*model.y[1]+
             3*model.y[2]+
             -0.1*model.y[4]+
             0.5*model.y[6])
       )
   model.ConstraintYmin = pyo.Constraint(
       expr = (model.x[1]+
               model.y[1]+
               model.y[2]+
               model.y[6] >= 0.25
               )
       )
   model.mysos = pyo.SOSConstraint(
       var=model.y, 
       sos=N
       )
   solver_name = 'scip'
   solver_available = bool(check_available_solvers(solver_name))
   if solver_available:
        opt = pyo.SolverFactory(solver_name)
        opt.solve(model, tee=False)
        assert isclose(pyo.value(model.OBJ), 0.05, abs_tol=1e-3)
