from pyomo.core.expr.sympy_tools import sympy2pyomo_expression, sympyify_expression
from pyomo.core.expr.numeric_expr import NumericExpression
from pyomo.core.expr.numvalue import is_fixed, value


def simplify_with_sympy(expr: NumericExpression):
    om, se = sympyify_expression(expr)
    se = se.simplify()
    new_expr = sympy2pyomo_expression(se, om)
    if is_fixed(new_expr):
        new_expr = value(new_expr)
    return new_expr