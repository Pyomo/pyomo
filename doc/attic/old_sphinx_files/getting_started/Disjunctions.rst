.. _disjunctstart:

Disjunctions
============

This is an advanced topic. See also :ref:`gdpchapt`.

A disjunction is a set of collections of variables, parameters, and constraints that are linked by an OR (really exclusive or) constraint. The simplest case is a 2-term disjunction:

D1 V D2

That is, either the constraints in the collection D1 are enforced, OR the constraints in the collection D2 are enforced.

In pyomo, we model each collection using a special type of block
called a ``Disjunct``. Each ``Disjunct`` is a block that contains an
implicitly declared binary variable, "indicator_var" that is 1 when
the constraints in that ``Disjunct`` is enforced and 0 otherwise.

Declaration
-----------

The following
condensed code snippet illustrates a ``Disjunct`` and a ``Disjunction``:

.. literalinclude:: spyfiles/spy4Disjunctions_Disjunct_and_disjunction.spy
   :language: python

Model.d is an indexed ``Disjunct`` that is indexed over an implicit set
with members 0 and 1. Since it is an indexed thing, each member is
initialized using a call to a rule, passing in the index value (just
like any other pyomo component). However, just defining disjuncts is
not sufficient to define disjunctions, as pyomo has no way of knowing
which disjuncts should be bundled into which disjunctions. To define a
disjunction, you use a ``Disjunction`` component. The disjunction takes
either a rule or an expression that returns a list of disjuncts over
which it should form the disjunction. This is what ``_c`` function in
the example returns.

NOTE: There is no requirement that disjuncts be indexed and also
no requirement that they be defined using a shared rule. It was
done in this case to create a condensed example.

Transformation
--------------

In order to use the solvers currently available, one must convert the
disjunctive model to a standard MIP/MINLP model.  The easiest way to
do that is using the (included) BigM or Convex Hull transformations.
From the Pyomo command line, include the option ``--transform pyomo.gdp.bigm``
or ``--transform pyomo.gdp.chull``

Notes
-----

Some notes:

.. autosummary::
   :nosignatures:

- all variables that appear in disjuncts need upper and lower bounds

- for linear models, the BigM transform can estimate reasonably tight M values for you

- for all other models, you will need to provide the M values through a “BigM” Suffix.

- the convex hull reformulation is only valid for linear and convex nonlinear problems.  Nonconvex problems are not supported (and are not checked for).

When you declare a Disjunct, it (at declaration time) will
automatically have a variable “indicator_var” defined and attached to
it.  After that, it is just a Var like any other Var.
