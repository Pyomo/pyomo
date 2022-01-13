import pyomo.environ as pe
import math
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging


logger = logging.getLogger(__name__)


def _get_longest_name(comps):
    longest_name = 0

    for i in comps:
        i_len = len(str(i))
        if i_len > longest_name:
            longest_name = i_len

    if longest_name > 195:
        longest_name = 195
    if longest_name < 12:
        longest_name = 12

    return longest_name


def _print_var_set(var_set):
    longest_varname = _get_longest_name(var_set)

    s = f'{"Var":<{longest_varname + 5}}{"LB":>12}{"UB":>12}\n'

    for v in var_set:
        if v.lb is None:
            v_lb = -math.inf
        else:
            v_lb = v.lb
        if v.ub is None:
            v_ub = math.inf
        else:
            v_ub = v.ub
        s += f'{str(v):<{longest_varname + 5}}{v_lb:>12.2e}{v_ub:>12.2e}\n'

    s += '\n'

    return s


def _check_var_bounds(m: _BlockData, too_large: float):
    vars_without_bounds = ComponentSet()
    vars_with_large_bounds = ComponentSet()
    for v in m.component_data_objects(pe.Var, descend_into=True):
        if v.is_fixed():
            continue
        if v.lb is None or v.ub is None:
            vars_without_bounds.add(v)
        elif v.lb <= -too_large or v.ub >= too_large:
            vars_with_large_bounds.add(v)

    return vars_without_bounds, vars_with_large_bounds


def _print_coefficients(comp_map):
    s = ''
    for c, der_bounds in comp_map.items():
        s += str(c)
        s += '\n'
        longest_vname = _get_longest_name([i[0] for i in der_bounds])
        s += f'    {"Var":<{longest_vname + 5}}{"Coef LB":>12}{"Coef UB":>12}\n'
        for v, der_lb, der_ub in der_bounds:
            s += f'    {str(v):<{longest_vname + 5}}{der_lb:>12.2e}{der_ub:>12.2e}\n'
        s += '\n'
    return s


def _check_coefficents(comp, expr, too_large, too_small, largs_coef_map, small_coef_map):
    ders = reverse_sd(expr)
    for _v, _der in ders.items():
        if isinstance(_v, _GeneralVarData):
            der_lb, der_ub = compute_bounds_on_expr(_der)
            if der_lb is None:
                der_lb = -math.inf
            if der_ub is None:
                der_ub = math.inf
            if der_lb <= -too_large or der_ub >= too_large:
                if comp not in largs_coef_map:
                    largs_coef_map[comp] = list()
                largs_coef_map[comp].append((_v, der_lb, der_ub))
            if abs(der_lb) <= too_small and abs(der_ub) < too_small:
                if comp not in small_coef_map:
                    small_coef_map[comp] = list()
                small_coef_map[comp].append((_v, der_lb, der_ub))


def report_scaling(m: _BlockData, too_large: float = 5e4, too_small: float = 1e-6) -> bool:
    """
    This function logs potentially poorly scaled parts of the model.
    It requires that all variables be bounded.

    It is important to note that this check is neither necessary nor sufficient
    to ensure a well-scaled model. However, it is a useful tool to help identify
    problematic parts of a model.

    This function uses symbolic differentiation and interval arithmetic
    to compute bounds on each entry in the jacobian of the constraints.

    Note that logging has to be turned on to get the output

    Parameters
    ----------
    m: _BlockData
        The pyomo model or block
    too_large: float
        Values above too_large will generate a log entry
    too_small: float
        Coefficients below too_small will generate a log entry

    Returns
    -------
    success: bool
        Returns False if any potentially poorly scaled components were found
    """
    vars_without_bounds, vars_with_large_bounds = _check_var_bounds(m, too_large)

    cons_with_large_bounds = dict()
    cons_with_large_coefficients = dict()
    cons_with_small_coefficients = dict()

    objs_with_large_coefficients = pe.ComponentMap()
    objs_with_small_coefficients = pe.ComponentMap()

    for c in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        _check_coefficents(c, c.body, too_large, too_small, cons_with_large_coefficients, cons_with_small_coefficients)

    for c in m.component_data_objects(pe.Constraint, active=True, descend_into=True):
        c_lb, c_ub = compute_bounds_on_expr(c.body)
        if c_lb is None:
            c_lb = -math.inf
        if c_ub is None:
            c_ub = math.inf
        if c_lb <= -too_large or c_ub >= too_large:
            cons_with_large_bounds[c] = (c_lb, c_ub)

    for c in m.component_data_objects(pe.Objective, active=True, descend_into=True):
        _check_coefficents(c, c.expr, too_large, too_small, objs_with_large_coefficients, objs_with_small_coefficients)

    s = '\n\n'

    if len(vars_without_bounds) > 0:
        s += 'The following variables are not bounded. Please add bounds.\n'
        s += _print_var_set(vars_without_bounds)

    if len(vars_with_large_bounds) > 0:
        s += 'The following variables have large bounds. Please scale them.\n'
        s += _print_var_set(vars_with_large_bounds)

    if len(objs_with_large_coefficients) > 0:
        s += 'The following objectives have potentially large coefficients. Please scale them.\n'
        s += _print_coefficients(objs_with_large_coefficients)

    if len(objs_with_small_coefficients) > 0:
        s += 'The following objectives have small coefficients.\n'
        s += _print_coefficients(objs_with_small_coefficients)

    if len(cons_with_large_coefficients) > 0:
        s += 'The following constraints have potentially large coefficients. Please scale them.\n'
        s += _print_coefficients(cons_with_large_coefficients)

    if len(cons_with_small_coefficients) > 0:
        s += 'The following constraints have small coefficients.\n'
        s += _print_coefficients(cons_with_small_coefficients)

    if len(cons_with_large_bounds) > 0:
        s += 'The following constraints have bodies with large bounds. Please scale them.\n'
        longest_cname = _get_longest_name(cons_with_large_bounds)
        s += f'{"Constraint":<{longest_cname + 5}}{"LB":>12}{"UB":>12}\n'
        for c, (c_lb, c_ub) in cons_with_large_bounds.items():
            s += f'{str(c):<{longest_cname + 5}}{c_lb:>12.2e}{c_ub:>12.2e}\n'

    if (len(vars_without_bounds) > 0
            or len(vars_with_large_bounds) > 0
            or len(cons_with_large_coefficients) > 0
            or len(cons_with_small_coefficients) > 0
            or len(objs_with_small_coefficients) > 0
            or len(objs_with_large_coefficients) > 0
            or len(cons_with_large_bounds) > 0):
        logger.info(s)
        return False
    return True
