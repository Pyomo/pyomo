.. |p| raw:: html

   <p />

Managing Expressions
====================

Cloning Expressions
-------------------

Expressions are automatically cloned only during certain expression
transformations.  Since this can be an expensive operation the
:class:`clone_counter_context <pyomo.core.expr.current.clone_counter_context>` context manager
is provided to track the number of times the :func:`clone_expression <pyomo.core.expr.current.clone_expression>`
function is executed.

Examples
~~~~~~~~

TODO

Source Documentation
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyomo.core.expr.current.clone_counter_context
    :members:

.. autofunction:: pyomo.core.expr.current.clone_expression

Walking an Expression Tree with a Visitor Class
-----------------------------------------------

TODO

Examples
~~~~~~~~

TODO

Source Documentation
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyomo.core.expr.current.SimpleExpressionVisitor
    :members:

.. autoclass:: pyomo.core.expr.current.ExpressionValueVisitor
    :members:

.. autoclass:: pyomo.core.expr.current.ExpressionReplacementVisitor
    :members:

