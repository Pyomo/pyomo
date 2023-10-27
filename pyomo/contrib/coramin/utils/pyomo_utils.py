import pyomo.environ as pe
from pyomo.core.expr.sympy_tools import sympyify_expression, sympy2pyomo_expression
from pyomo.core.expr.numvalue import is_fixed
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables


def get_objective(m):
    """
    Assert that there is only one active objective in m and return it.

    Parameters
    ----------
    m: pyomo.core.base.block._BlockData

    Returns
    -------
    obj: pyomo.core.base.objective._ObjectiveData
    """
    obj = None
    for o in m.component_data_objects(pe.Objective, descend_into=True, active=True, sort=True):
        if obj is not None:
            raise ValueError('Found multiple active objectives')
        obj = o
    return obj


def unfixed_vars(m, descend_into=True, active=True):
    for v in m.component_data_objects(pe.Var, descend_into=descend_into, active=active):
        if not v.is_fixed():
            yield v


def active_vars(m, include_fixed=False):
    seen = set()
    for c in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        for v in identify_variables(c.body, include_fixed=include_fixed):
            v_id = id(v)
            if v_id not in seen:
                seen.add(v_id)
                yield v
    obj = get_objective(m)
    if obj is not None:
        for v in identify_variables(obj.expr, include_fixed=include_fixed):
            v_id = id(v)
            if v_id not in seen:
                seen.add(v_id)
                yield v


def simplify_expr(expr):
    om, se = sympyify_expression(expr)
    se = se.simplify()
    new_expr = sympy2pyomo_expression(se, om)
    if is_fixed(new_expr):
        new_expr = pe.value(new_expr)
    return new_expr
