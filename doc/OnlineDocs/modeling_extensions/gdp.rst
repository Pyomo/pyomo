Generalized Disjunctive Programming
___________________________________

.. image:: /../logos/gdp/Pyomo-GDP-150.png
   :scale: 35%
   :align: right

The Pyomo.GDP modeling extension allows users to include logical disjunctions in their models.
These disjunctions are often used to model discrete decisions that have implications on the system behavior.
For example, in process design, a disjunction may model the choice between processes A and B.
If A is selected, then its associated equations and inequalities will apply; otherwise, if B is selected,
then its respective constraints should be enforced.
In the general case, if these models contain nonlinear relations,
then they are Generalized Disjunctive Programming (GDP) models

Disjunctions
============

A disjunction is a set of constraint groupings that are linked by a logical OR relationship.
The simplest case is a 2-term disjunction:

.. math::

   D_1 \vee D_2

That is, either the constraints in the collection D\ :sub:`1` are
enforced, OR the constraints in the collection D\ :sub:`2` are enforced.

In Pyomo, we model each collection using a special type of block called
a ``Disjunct``. Each ``Disjunct`` is a block that contains an implicitly
declared binary variable, "indicator_var" that is 1 when the constraints
in that ``Disjunct`` is enforced and 0 otherwise.

Declaration
-----------

The following condensed code snippet illustrates a ``Disjunct`` and a
``Disjunction``:

.. literalinclude:: ../script_spy_files/spy4Disjunctions_Disjunct_and_disjunction.spy
   :language: python

Model.d is an indexed ``Disjunct`` that is indexed over an implicit set
with members 0 and 1. Since it is an indexed thing, each member is
initialized using a call to a rule, passing in the index value (just
like any other pyomo component). However, just defining disjuncts is not
sufficient to define disjunctions, as pyomo has no way of knowing which
disjuncts should be bundled into which disjunctions. To define a
disjunction, you use a ``Disjunction`` component. The disjunction takes
either a rule or an expression that returns a list of disjuncts over
which it should form the disjunction. This is what ``_c`` function in
the example returns.

.. note::

   There is no requirement that disjuncts be indexed and also no
   requirement that they be defined using a shared rule. It was done in
   this case to create a condensed example.

Transformation
--------------

To use standard commercial solvers, you must convert the disjunctive model to a standard MIP/MINLP model.
The two classical strategies for doing so are the (included) Big-M and Hull reformulations.
From the Pyomo command line,  include the option ``--transform pyomo.gdp.bigm`` or ``--transform pyomo.gdp.chull``.
If you are using a Python script, ``TransformationFactory`` accomplishes the same functionality:

- ``TransformationFactory('gdp.bigm').apply_to(model)``
- ``TransformationFactory('gdp.chull').apply_to(model)``

.. note::

    - all variables that appear in disjuncts need upper and lower bounds for chull

    - for linear models, the BigM transform can estimate reasonably tight M
      values for you if variables are bounded.

    - for nonlinear models where finite expression bounds may be inferred from
      variable bounds, the BigM transformation may also be able to automatically
      compute M values for you.

    - for all other models, you will need to provide the M values through a
      “BigM” Suffix. A ``GDP_Error`` will be raised for missing M values.

    - When you declare a Disjunct, it (at declaration time) will automatically
      have a variable “indicator_var” defined and attached to it.
      After that, it is just a Var like any other Var.

    - The hull reformulation is an exact reformulation at the solution points
      even for nonconvex models, but the resulting MINLP will also be nonconvex.

Direct GDP solvers
------------------

Pyomo includes the contributed GDPopt solver, which can direct solve GDP models.
Its documentation and usage is described at :doc:`/contributed_packages/gdpopt`.

Examples
--------

The following models all work and are equivalent:

.. doctest::
   
   Option 1: maximal verbosity, abstract-like

   >>> from pyomo.environ import *
   >>> from pyomo.gdp import *
   >>> model = ConcreteModel()

   >>> model.x = Var()
   >>> model.y = Var()

   >>> # Two conditions
   >>> def _d(disjunct, flag):
   ...    model = disjunct.model()
   ...    if flag:
   ...       # x == 0
   ...       disjunct.c = Constraint(expr=model.x == 0)
   ...    else:
   ...       # y == 0
   ...       disjunct.c = Constraint(expr=model.y == 0)
   >>> model.d = Disjunct([0,1], rule=_d)
 
   >>> # Define the disjunction
   >>> def _c(model):
   ...    return [model.d[0], model.d[1]]
   >>> model.c = Disjunction(rule=_c) 

   Option 2: Maximal verbosity, concrete-like:

   >>> from pyomo.environ import *
   >>> from pyomo.gdp import *
   >>> model = ConcreteModel()
 
   >>> model.x = Var()
   >>> model.y = Var()
 
   >>> model.fix_x = Disjunct()
   >>> model.fix_x.c = Constraint(expr=model.x == 0)
 
   >>> model.fix_y = Disjunct()
   >>> model.fix_y.c = Constraint(expr=model.y == 0)
 
   >>> model.c = Disjunction(expr=[model.fix_x, model.fix_y])
 
   Option 3: Implicit disjuncts (disjunction rule returns a list of
   expressions or a list of lists of expressions)
 
   >>> from pyomo.environ import *
   >>> from pyomo.gdp import *
   >>> model = ConcreteModel()
 
   >>> model.x = Var()
   >>> model.y = Var()

   >>> model.c = Disjunction(expr=[model.x == 0, model.y == 0])


