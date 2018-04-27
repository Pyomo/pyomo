.. |p| raw:: html

   <p />

Pyomo Expressions
=================

.. warning::

    This documentation does not explicitly reference objects in
    pyomo.core.kernel.  While the Pyomo5 expression system works
    with pyomo.core.kernel objects, the documentation of these
    documents was not sufficient to appropriately descibe the use
    of kernel objects in expressions.

Pyomo supports the declaration of symbolic expressions that represent
objectives, constraints and other optimization modeling components.
Pyomo expressions are represented in an expression tree, where the
leaves are operands, such as constants or variables, and the internal
nodes contain operators.  Pyomo relies on so-called magic methods
to automate the construction of symbolic expressions.  For example,
consider an expression ``e`` declared as follows:

.. literalinclude:: ../../tests/expr/index_simple.spy

Python determines that the magic method ``__mul__`` is called on
the ``M.v`` object, with the argument ``2``.  This method returns
a Pyomo expression object ``ProductExpression`` that has arguments
``M.v`` and ``2``.  This represents the following symbolic expression
tree:

.. graphviz::

    digraph foo {
        "*" -> "v";
        "*" -> "2";
    }

.. note::

    End-users will not likely need to know details related to how
    symbolic expressions are generated and managed in Pyomo.  Thus,
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

