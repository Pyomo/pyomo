Core Classes
============

.. currentmodule:: pyomo.core.expr.numeric_expr

The following are the two core classes documented here:

   * :class:`NumericValue`
   * :class:`NumericExpression`

The remaining classes are the public classes for expressions, which
developers may need to know about. The methods for these classes are not
documented because they are described in the
:class:`NumericExpression` class.

Sets with Expression Types
--------------------------

The following sets can be used to develop visitor patterns for 
Pyomo expressions.

.. autosummary::

   ~pyomo.common.numeric_types.native_numeric_types
   ~pyomo.common.numeric_types.native_types
   ~pyomo.common.numeric_types.nonpyomo_leaf_types

NumericValue and NumericExpression
----------------------------------

.. autosummary::

   NumericValue
   NumericExpression

Other Public Classes
--------------------


.. autosummary::

   NegationExpression
   AbsExpression
   UnaryFunctionExpression
   ProductExpression
   DivisionExpression
   SumExpression
   Expr_ifExpression
   ExternalFunctionExpression
   pyomo.core.expr.relational_expr.EqualityExpression
   pyomo.core.expr.relational_expr.InequalityExpression
   pyomo.core.expr.relational_expr.RangedExpression
   pyomo.core.expr.template_expr.GetItemExpression
