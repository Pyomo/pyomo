.. |p| raw:: html

   <p />

Pyomo Expressions
=================

.. warning::

    This is draft documentation for Pyomo expressions.  In particular,
    this documentation describes Pyomo5 expressions in detail.  The 
    API for the functions and classes being described here is being
    finalized.  Please submit a ticket to suggest a change in the
    API or to request clarification in the design below.

.. warning::

    The examples in this draft documentation are not well-tested.
    The examples look fine to the author, but I have not confirmed that these 
    examples are tested with Pymo.

Pyomo supports the declaration of abstract expressions that represent
objectives, constraints and other optimization modeling components.
Pyomo expressions are represented in an expression tree, where the
leaves are operands, such as constants or variables, and the internal
nodes contain operators.  Pyomo relies on so-called magic methods
to automate the construction of abstract expressions.  For example,
consider an expression ``e`` declared as follows:

.. doctest::

   >>> from pyomo.environ import *

   >>> M = ConcreteModel()
   >>> M.v = Var()

   >>> e = M.v*2

Python determines that the magic method ``__mul__`` is called on
the ``M.v`` object, with the argument ``2``.  This method returns
a Pyomo expression object ``_ProductExpression`` that has arguments
``M.v`` and ``2``.  This represents the following abstract expression
tree:

.. graphviz::

    digraph foo {
        "*" -> "v";
        "*" -> "2";
    }

.. note::

    End-users will not likely need to know details related to how
    abstract expressions are generated and managed in Pyomo.  Thus,
    most of the following documentation of expressions in Pyomo is most
    useful for Pyomo developers.  However, the discussion of runtime
    performance in the first section will help end-users write large-scale
    models.

.. toctree::
   :maxdepth: 1

   performance.rst
   overview.rst 
   design.rst
   managing.rst

