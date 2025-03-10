#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
import math
from pyomo.core.base.block import BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import Var
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging


logger = logging.getLogger(__name__)


def _bounds_to_float(lb, ub):
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf
    return lb, ub


def _print_var_set(var_set):
    s = f'{"LB":>12}{"UB":>12}    Var\n'

    for v in var_set:
        v_lb, v_ub = _bounds_to_float(*v.bounds)
        s += f'{v_lb:>12.2e}{v_ub:>12.2e}    {str(v)}\n'

    s += '\n'

    return s


def _check_var_bounds(m: BlockData, too_large: float):
    vars_without_bounds = ComponentSet()
    vars_with_large_bounds = ComponentSet()
    for v in m.component_data_objects(pyo.Var, descend_into=True):
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
        s += f'    {"Coef LB":>12}{"Coef UB":>12}    Var\n'
        for v, der_lb, der_ub in der_bounds:
            s += f'    {der_lb:>12.2e}{der_ub:>12.2e}    {str(v)}\n'
        s += '\n'
    return s


def _check_coefficients(
    comp, expr, too_large, too_small, largs_coef_map, small_coef_map
):
    ders = reverse_sd(expr)
    for _v, _der in ders.items():
        if getattr(_v, 'ctype', None) is Var:
            if _v.is_fixed():
                continue
            der_lb, der_ub = compute_bounds_on_expr(_der)
            der_lb, der_ub = _bounds_to_float(der_lb, der_ub)
            if der_lb <= -too_large or der_ub >= too_large:
                if comp not in largs_coef_map:
                    largs_coef_map[comp] = list()
                largs_coef_map[comp].append((_v, der_lb, der_ub))
            if abs(der_lb) <= too_small and abs(der_ub) < too_small:
                if der_lb != 0 or der_ub != 0:
                    if comp not in small_coef_map:
                        small_coef_map[comp] = list()
                    small_coef_map[comp].append((_v, der_lb, der_ub))


def report_scaling(
    m: BlockData, too_large: float = 5e4, too_small: float = 1e-6
) -> bool:
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
    m: BlockData
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

    objs_with_large_coefficients = pyo.ComponentMap()
    objs_with_small_coefficients = pyo.ComponentMap()

    for c in m.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        _check_coefficients(
            c,
            c.body,
            too_large,
            too_small,
            cons_with_large_coefficients,
            cons_with_small_coefficients,
        )

    for c in m.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        c_lb, c_ub = compute_bounds_on_expr(c.body)
        c_lb, c_ub = _bounds_to_float(c_lb, c_ub)
        if c_lb <= -too_large or c_ub >= too_large:
            cons_with_large_bounds[c] = (c_lb, c_ub)

    for c in m.component_data_objects(pyo.Objective, active=True, descend_into=True):
        _check_coefficients(
            c,
            c.expr,
            too_large,
            too_small,
            objs_with_large_coefficients,
            objs_with_small_coefficients,
        )

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
        s += f'{"LB":>12}{"UB":>12}    Constraint\n'
        for c, (c_lb, c_ub) in cons_with_large_bounds.items():
            s += f'{c_lb:>12.2e}{c_ub:>12.2e}    {str(c)}\n'

    if (
        len(vars_without_bounds) > 0
        or len(vars_with_large_bounds) > 0
        or len(cons_with_large_coefficients) > 0
        or len(cons_with_small_coefficients) > 0
        or len(objs_with_small_coefficients) > 0
        or len(objs_with_large_coefficients) > 0
        or len(cons_with_large_bounds) > 0
    ):
        logger.info(s)
        return False
    return True
