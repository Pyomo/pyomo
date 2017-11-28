
Reference Documentation
=======================

Utilities to Build Expressions
------------------------------

.. autofunction:: pyomo.core.util.prod
.. autofunction:: pyomo.core.util.Sum
.. autofunction:: pyomo.core.util.summation
.. autodata::     pyomo.core.util.dot_product

Utilities to Manage and Analyze Expressions
-------------------------------------------

.. autofunction:: pyomo.core.expr.current.clone_expression
.. autofunction:: pyomo.core.expr.current.evaluate_expression
.. autofunction:: pyomo.core.expr.current.identify_components
.. autofunction:: pyomo.core.expr.current.identify_variables

Context Managers
----------------

.. autodata:: pyomo.core.expr.current.nonlinear_expression
.. autoclass:: pyomo.core.expr.current.mutable_sum_context
    :members:

.. autodata:: pyomo.core.expr.current.linear_expression
.. autoclass:: pyomo.core.expr.current.mutable_linear_context
    :members:

.. autodata:: pyomo.core.expr.current.clone_counter
.. autoclass:: pyomo.core.expr.current.clone_counter_context
    :members:


Visitor Classes
---------------

.. autoclass:: pyomo.core.expr.current.SimpleExpressionVisitor
    :members:
.. autoclass:: pyomo.core.expr.current.ExpressionValueVisitor
    :members:
.. autoclass:: pyomo.core.expr.current.ExpressionReplacementVisitor
    :members:

