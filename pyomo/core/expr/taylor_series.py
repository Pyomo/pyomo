from pyomo.core.expr.current import identify_variables, value
from pyomo.core.expr.differentiate.differentiate import differentiate, DiffModes


def taylor_series(expr, diff_mode=DiffModes.reverse_numeric):
    """
    Generate a taylor series approximation for expr.

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.ExpressionBase
    diff_mode: pyomo.core.expr.differentiate.differentiate.DiffModes
        The method for differentiation.

    Returns
    -------
    res: pyomo.core.expr.numeric_expr.ExpressionBase
    """
    e_vars = list(identify_variables(expr=expr, include_fixed=False))
    derivs = differentiate(expr=expr, wrt_list=e_vars, mode=diff_mode)
    res = value(expr) + sum(value(derivs[i]) * (e_vars[i] - e_vars[i].value) for i in range(len(e_vars)))
    return res
