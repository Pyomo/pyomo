Core Classes
============

The following are the two core classes documented here:

   * :class:`NumericValue<pyomo.core.expr.numvalue.NumericValue>`
   * :class:`ExpressionBase<pyomo.core.expr.current.ExpressionBase>`

The remaining classes are the public classes for expressions, which
developers may need to know about. The methods for these classes are not
documented because they are described in the
:class:`ExpressionBase<pyomo.core.expr.current.ExpressionBase>` class.

Sets with Expression Types
--------------------------

The following sets can be used to develop visitor patterns for 
Pyomo expressions.

.. autodata:: pyomo.core.expr.numvalue.native_numeric_types
.. autodata:: pyomo.core.expr.numvalue.native_types
.. autodata:: pyomo.core.expr.numvalue.nonpyomo_leaf_types

NumericValue and ExpressionBase
-------------------------------

.. autoclass:: pyomo.core.expr.numvalue.NumericValue
    :members:
    :special-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.ExpressionBase
    :members:
    :show-inheritance:
    :special-members:
    :private-members:

Other Public Classes
--------------------

.. autoclass:: pyomo.core.expr.current.NegationExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.ExternalFunctionExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.ProductExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.ReciprocalExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.InequalityExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.EqualityExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.SumExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.GetItemExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.Expr_ifExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.UnaryFunctionExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.current.AbsExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:
