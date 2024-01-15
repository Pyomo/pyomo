Core Classes
============

The following are the two core classes documented here:

   * :class:`NumericValue<pyomo.core.expr.numvalue.NumericValue>`
   * :class:`NumericExpression<pyomo.core.expr.NumericExpression>`

The remaining classes are the public classes for expressions, which
developers may need to know about. The methods for these classes are not
documented because they are described in the
:class:`NumericExpression<pyomo.core.expr.NumericExpression>` class.

Sets with Expression Types
--------------------------

The following sets can be used to develop visitor patterns for 
Pyomo expressions.

.. autodata:: pyomo.core.expr.numvalue.native_numeric_types
.. autodata:: pyomo.core.expr.numvalue.native_types
.. autodata:: pyomo.core.expr.numvalue.nonpyomo_leaf_types

NumericValue and NumericExpression
----------------------------------

.. autoclass:: pyomo.core.expr.numvalue.NumericValue
    :members:
    :special-members:
    :private-members:

.. autoclass:: pyomo.core.expr.NumericExpression
    :members:
    :show-inheritance:
    :special-members:
    :private-members:

Other Public Classes
--------------------

.. autoclass:: pyomo.core.expr.NegationExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.ExternalFunctionExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.ProductExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.DivisionExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.InequalityExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.EqualityExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.SumExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.GetItemExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.Expr_ifExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.UnaryFunctionExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:

.. autoclass:: pyomo.core.expr.AbsExpression
    :members:
    :show-inheritance:
    :undoc-members:
    :private-members:
