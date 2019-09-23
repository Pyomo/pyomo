LinearExpression
================

Significant speed
improvements can be obtained using the ``LinearExpression`` object
when there are long, dense, linear expressions. The arguments are

:: 

   constant, linear_coeffs, linear_vars

where the second and third arguments are lists that must be of the
same length. Here is a simple example that illustrates the
syntax. This example creates two constraints that are the same:

.. doctest::

   >>> import pyomo.environ as pyo
   >>> from pyomo.core.expr.numeric_expr import LinearExpression
   >>> model = pyo.ConcreteModel()
   >>> model.nVars = pyo.Param(initialize=4)
   >>> model.N = pyo.RangeSet(model.nVars)
   >>> model.x = pyo.Var(model.N, within=pyo.Binary)
   >>> 
   >>> model.coefs = [1, 1, 3, 4]
   >>> 
   >>> model.linexp = LinearExpression(constant=0,
   ...                                 linear_coefs=model.coefs,
   ...                                 linear_vars=[model.x[i] for i in model.N])
   >>> def caprule(m):
   ...     return m.linexp <= 6
   >>> model.capme = pyo.Constraint(rule=caprule)
   >>>
   >>> def caprule2(m):
   ...     return sum(model.coefs[i-1]*model.x[i] for i in model.N) <= 6
   >>> model.capme2 = pyo.Constraint(rule=caprule2)
   

.. warning::

   The lists that are passed to ``LinearModel`` are not copied, so caution must
   be excercised if they are modified after the component is constructed.
