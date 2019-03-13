from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
import pyomo.core.expr.expr_pyomo5 as _expr
from pyomo.core.expr.expr_pyomo5 import (ExpressionValueVisitor,
                                         nonpyomo_leaf_types, value,
                                         identify_variables)
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
import logging

logger = logging.getLogger(__name__)

if not hasattr(math, 'inf'):
    math.inf = float('inf')


"""
The purpose of this file is to perform feasibility based bounds 
tightening. This is a very basic implementation, but it is done 
directly with pyomo expressions. The only functions that are meant to 
be used by users are fbbt, fbbt_con, and fbbt_block. The first set of 
functions in this file (those with names starting with 
_prop_bnds_leaf_to_root) are used for propagating bounds from the  
variables to each node in the expression tree (all the way to the  
root node). The second set of functions (those with names starting 
with _prop_bnds_root_to_leaf) are used to propagate bounds from the 
constraint back to the variables. For example, consider the constraint 
x*y + z == 1 with -1 <= x <= 1 and -2 <= y <= 2. When propagating 
bounds from the variables to the root (the root is x*y + z), we find 
that -2 <= x*y <= 2, and that -inf <= x*y + z <= inf. However, 
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
propagate bounds back to the variables. Since we know that 
1 <= x*y + z <= 1 and -2 <= x*y <= 2, then we must have -1 <= z <= 3. 
However, bounds cannot be improved on x*y, so bounds cannot be 
improved on either x or y.

>>> import pyomo.environ as pe
>>> m = pe.ConcreteModel()
>>> m.x = pe.Var(bounds=(-1,1))
>>> m.y = pe.Var(bounds=(-2,2))
>>> m.z = pe.Var()
>>> from pyomo.contrib.fbbt.fbbt import fbbt
>>> m.c = pe.Constraint(expr=m.x*m.y + m.z == 1)
>>> fbbt(m)
>>> print(m.z.lb, m.z.ub)
-1.0 3.0

"""


class FBBTException(Exception):
    pass


def _prop_bnds_leaf_to_root_ProductExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    bnds_dict[node] = interval.mul(lb1, ub1, lb2, ub2)


def _prop_bnds_leaf_to_root_SumExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.SumExpression
    bnds_dict: ComponentMap
    """
    arg0 = node.arg(0)
    lb, ub = bnds_dict[arg0]
    for i in range(1, node.nargs()):
        arg = node.arg(i)
        lb2, ub2 = bnds_dict[arg]
        lb, ub = interval.add(lb, ub, lb2, ub2)
    bnds_dict[node] = (lb, ub)


def _prop_bnds_leaf_to_root_PowExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.PowExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    bnds_dict[node] = interval.power(lb1, ub1, lb2, ub2)


def _prop_bnds_leaf_to_root_ReciprocalExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ReciprocalExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.inv(lb1, ub1)


def _prop_bnds_leaf_to_root_NegationExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.sub(0, 0, lb1, ub1)


def _prop_bnds_leaf_to_root_exp(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.exp(lb1, ub1)


def _prop_bnds_leaf_to_root_log(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.log(lb1, ub1)


def _prop_bnds_leaf_to_root_sin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.sin(lb1, ub1)


def _prop_bnds_leaf_to_root_cos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.cos(lb1, ub1)


def _prop_bnds_leaf_to_root_tan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.tan(lb1, ub1)


def _prop_bnds_leaf_to_root_asin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.asin(lb1, ub1, -math.inf, math.inf)


def _prop_bnds_leaf_to_root_acos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.acos(lb1, ub1, -math.inf, math.inf)


def _prop_bnds_leaf_to_root_atan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.atan(lb1, ub1, -math.inf, math.inf)


_unary_leaf_to_root_map = dict()
_unary_leaf_to_root_map['exp'] = _prop_bnds_leaf_to_root_exp
_unary_leaf_to_root_map['log'] = _prop_bnds_leaf_to_root_log
_unary_leaf_to_root_map['sin'] = _prop_bnds_leaf_to_root_sin
_unary_leaf_to_root_map['cos'] = _prop_bnds_leaf_to_root_cos
_unary_leaf_to_root_map['tan'] = _prop_bnds_leaf_to_root_tan
_unary_leaf_to_root_map['asin'] = _prop_bnds_leaf_to_root_asin
_unary_leaf_to_root_map['acos'] = _prop_bnds_leaf_to_root_acos
_unary_leaf_to_root_map['atan'] = _prop_bnds_leaf_to_root_atan


def _prop_bnds_leaf_to_root_UnaryFunctionExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    if node.getname() in _unary_leaf_to_root_map:
        _unary_leaf_to_root_map[node.getname()](node, bnds_dict)
    else:
        bnds_dict[node] = (-math.inf, math.inf)


_prop_bnds_leaf_to_root_map = dict()
_prop_bnds_leaf_to_root_map[_expr.ProductExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[_expr.ReciprocalExpression] = _prop_bnds_leaf_to_root_ReciprocalExpression
_prop_bnds_leaf_to_root_map[_expr.PowExpression] = _prop_bnds_leaf_to_root_PowExpression
_prop_bnds_leaf_to_root_map[_expr.SumExpression] = _prop_bnds_leaf_to_root_SumExpression
_prop_bnds_leaf_to_root_map[_expr.MonomialTermExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[_expr.NegationExpression] = _prop_bnds_leaf_to_root_NegationExpression
_prop_bnds_leaf_to_root_map[_expr.UnaryFunctionExpression] = _prop_bnds_leaf_to_root_UnaryFunctionExpression

_prop_bnds_leaf_to_root_map[_expr.NPV_ProductExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[_expr.NPV_ReciprocalExpression] = _prop_bnds_leaf_to_root_ReciprocalExpression
_prop_bnds_leaf_to_root_map[_expr.NPV_PowExpression] = _prop_bnds_leaf_to_root_PowExpression
_prop_bnds_leaf_to_root_map[_expr.NPV_SumExpression] = _prop_bnds_leaf_to_root_SumExpression
_prop_bnds_leaf_to_root_map[_expr.NPV_NegationExpression] = _prop_bnds_leaf_to_root_NegationExpression
_prop_bnds_leaf_to_root_map[_expr.NPV_UnaryFunctionExpression] = _prop_bnds_leaf_to_root_UnaryFunctionExpression


def _prop_bnds_root_to_leaf_ProductExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    _lb1, _ub1 = interval.div(lb0, ub0, lb2, ub2)
    _lb2, _ub2 = interval.div(lb0, ub0, lb1, ub1)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    if _lb2 > lb2:
        lb2 = _lb2
    if _ub2 < ub2:
        ub2 = _ub2
    bnds_dict[arg1] = (lb1, ub1)
    bnds_dict[arg2] = (lb2, ub2)


def _prop_bnds_root_to_leaf_SumExpression(node, bnds_dict):
    """
    This function is a bit complicated. A simpler implementation
    would loop through each argument in the sum and do the following:

    bounds_on_arg_i = bounds_on_entire_sum - bounds_on_sum_of_args_excluding_arg_i

    and the bounds_on_sum_of_args_excluding_arg_i could be computed
    for each argument. However, the computational expense would grow
    approximately quadratically with the length of the sum. Thus,
    we do the following. Consider the expression

    y = x1 + x2 + x3 + x4

    and suppose we have bounds on y. We first accumulate bounds to
    obtain a list like the following

    [(x1)_bounds, (x1+x2)_bounds, (x1+x2+x3)_bounds, (x1+x2+x3+x4)_bounds]

    Then we can propagate bounds back to x1, x2, x3, and x4 with the
    following

    (x4)_bounds = (x1+x2+x3+x4)_bounds - (x1+x2+x3)_bounds
    (x3)_bounds = (x1+x2+x3)_bounds - (x1+x2)_bounds
    (x2)_bounds = (x1+x2)_bounds - (x1)_bounds

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    # first accumulate bounds
    accumulated_bounds = list()
    accumulated_bounds.append(bnds_dict[node.arg(0)])
    lb0, ub0 = bnds_dict[node]
    for i in range(1, node.nargs()):
        _lb0, _ub0 = accumulated_bounds[i-1]
        _lb1, _ub1 = bnds_dict[node.arg(i)]
        accumulated_bounds.append(interval.add(_lb0, _ub0, _lb1, _ub1))
    if lb0 > accumulated_bounds[node.nargs() - 1][0]:
        accumulated_bounds[node.nargs() - 1] = (lb0, accumulated_bounds[node.nargs()-1][1])
    if ub0 < accumulated_bounds[node.nargs() - 1][1]:
        accumulated_bounds[node.nargs() - 1] = (accumulated_bounds[node.nargs()-1][0], ub0)

    for i in reversed(range(1, node.nargs())):
        lb0, ub0 = accumulated_bounds[i]
        lb1, ub1 = accumulated_bounds[i-1]
        lb2, ub2 = bnds_dict[node.arg(i)]
        _lb1, _ub1 = interval.sub(lb0, ub0, lb2, ub2)
        _lb2, _ub2 = interval.sub(lb0, ub0, lb1, ub1)
        if _lb1 > lb1:
            lb1 = _lb1
        if _ub1 < ub1:
            ub1 = _ub1
        if _lb2 > lb2:
            lb2 = _lb2
        if _ub2 < ub2:
            ub2 = _ub2
        accumulated_bounds[i-1] = (lb1, ub1)
        bnds_dict[node.arg(i)] = (lb2, ub2)
    lb, ub = bnds_dict[node.arg(0)]
    _lb, _ub = accumulated_bounds[0]
    if _lb > lb:
        lb = _lb
    if _ub < ub:
        ub = _ub
    bnds_dict[node.arg(0)] = (lb, ub)


def _prop_bnds_root_to_leaf_PowExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    _lb1a, _ub1a = interval.log(lb0, ub0)
    _lb1a, _ub1a = interval.div(_lb1a, _ub1a, lb2, ub2)
    _lb1a, _ub1a = interval.exp(_lb1a, _ub1a)
    _lb1b, _ub1b = interval.sub(0, 0, _lb1a, _ub1a)
    _lb1 = min(_lb1a, _lb1b)
    _ub1 = max(_ub1a, _ub1b)
    _lb2a, _ub2a = interval.log(lb0, ub0)
    _lb2b, _ub2b = interval.log(lb1, ub1)
    _lb2, _ub2 = interval.div(_lb2a, _ub2a, _lb2b, _ub2b)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    if _lb2 > lb2:
        lb2 = _lb2
    if _ub2 < ub2:
        ub2 = _ub2
    bnds_dict[arg1] = (lb1, ub1)
    bnds_dict[arg2] = (lb2, ub2)


def _prop_bnds_root_to_leaf_ReciprocalExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.inv(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_NegationExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.sub(0, 0, lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_exp(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.log(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_log(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.exp(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_sin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.asin(lb0, ub0, lb1, ub1)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_cos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.acos(lb0, ub0, lb1, ub1)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_tan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.atan(lb0, ub0, lb1, ub1)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_asin(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.sin(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_acos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.cos(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_atan(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.tan(lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


_unary_root_to_leaf_map = dict()
_unary_root_to_leaf_map['exp'] = _prop_bnds_root_to_leaf_exp
_unary_root_to_leaf_map['log'] = _prop_bnds_root_to_leaf_log
_unary_root_to_leaf_map['sin'] = _prop_bnds_root_to_leaf_sin
_unary_root_to_leaf_map['cos'] = _prop_bnds_root_to_leaf_cos
_unary_root_to_leaf_map['tan'] = _prop_bnds_root_to_leaf_tan
_unary_root_to_leaf_map['asin'] = _prop_bnds_root_to_leaf_asin
_unary_root_to_leaf_map['acos'] = _prop_bnds_root_to_leaf_acos
_unary_root_to_leaf_map['atan'] = _prop_bnds_root_to_leaf_atan


def _prop_bnds_root_to_leaf_UnaryFunctionExpression(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    if node.getname() in _unary_root_to_leaf_map:
        _unary_root_to_leaf_map[node.getname()](node, bnds_dict)
    else:
        logger.warning('Unsupported expression type for FBBT: {0}. Bounds will not be improved in this part of '
                       'the tree.'
                       ''.format(str(type(node))))


_prop_bnds_root_to_leaf_map = dict()
_prop_bnds_root_to_leaf_map[_expr.ProductExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[_expr.ReciprocalExpression] = _prop_bnds_root_to_leaf_ReciprocalExpression
_prop_bnds_root_to_leaf_map[_expr.PowExpression] = _prop_bnds_root_to_leaf_PowExpression
_prop_bnds_root_to_leaf_map[_expr.SumExpression] = _prop_bnds_root_to_leaf_SumExpression
_prop_bnds_root_to_leaf_map[_expr.MonomialTermExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[_expr.NegationExpression] = _prop_bnds_root_to_leaf_NegationExpression
_prop_bnds_root_to_leaf_map[_expr.UnaryFunctionExpression] = _prop_bnds_root_to_leaf_UnaryFunctionExpression

_prop_bnds_root_to_leaf_map[_expr.NPV_ProductExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[_expr.NPV_ReciprocalExpression] = _prop_bnds_root_to_leaf_ReciprocalExpression
_prop_bnds_root_to_leaf_map[_expr.NPV_PowExpression] = _prop_bnds_root_to_leaf_PowExpression
_prop_bnds_root_to_leaf_map[_expr.NPV_SumExpression] = _prop_bnds_root_to_leaf_SumExpression
_prop_bnds_root_to_leaf_map[_expr.NPV_NegationExpression] = _prop_bnds_root_to_leaf_NegationExpression
_prop_bnds_root_to_leaf_map[_expr.NPV_UnaryFunctionExpression] = _prop_bnds_root_to_leaf_UnaryFunctionExpression


class _FBBTVisitorLeafToRoot(ExpressionValueVisitor):
    """
    This walker propagates bounds from the variables to each node in
    the expression tree (all the way to the root node).
    """
    def __init__(self, bnds_dict, integer_tol=1e-4):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        integer_tol: float
        """
        self.bnds_dict = bnds_dict
        self.integer_tol = integer_tol

    def visit(self, node, values):
        if node.__class__ in _prop_bnds_leaf_to_root_map:
            _prop_bnds_leaf_to_root_map[node.__class__](node, self.bnds_dict)
        else:
            self.bnds_dict[node] = (-math.inf, math.inf)
        return None

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.bnds_dict[node] = (node, node)
            return True, None

        if node.is_variable_type():
            if node.is_fixed():
                lb = value(node.value)
                ub = lb
            else:
                lb = value(node.lb)
                ub = value(node.ub)
                if lb is None:
                    lb = -math.inf
                if ub is None:
                    ub = math.inf
            self.bnds_dict[node] = (lb, ub)
            return True, None

        if not node.is_expression_type():
            assert is_fixed(node)
            val = value(node)
            self.bnds_dict[node] = (val, val)
            return True, None

        return False, None


class _FBBTVisitorRootToLeaf(ExpressionValueVisitor):
    """
    This walker propagates bounds from the constraint back to the
    variables. Note that the bounds on every node in the tree must
    first be computed with _FBBTVisitorLeafToRoot.
    """
    def __init__(self, bnds_dict, update_variable_bounds=True, integer_tol=1e-4):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        update_variable_bounds: bool
        integer_tol: float
        """
        self.bnds_dict = bnds_dict
        self.update_var_bounds = update_variable_bounds
        self.integer_tol = integer_tol

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, None

        if node.is_variable_type():
            lb, ub = self.bnds_dict[node]

            if node.is_binary() or node.is_integer():
                """
                This bit of code has two purposes:
                1) Improve the bounds on binary and integer variables with the fact that they are integer.
                2) Account for roundoff error. If the lower bound of a binary variable comes back as 
                   1e-16, the lower bound may actually be 0. This could potentially cause problems when 
                   handing the problem to a MIP solver. Some solvers are robust to this, but some may not be
                   and may give the wrong solution. Even if the correct solution is found, this could 
                   introduce numerical problems.
                """
                lb = max(math.floor(lb), math.ceil(lb - self.integer_tol))
                ub = min(math.ceil(ub), math.floor(ub + self.integer_tol))
                if lb < value(node.lb):
                    lb = value(node.lb)  # don't make the bounds worse than the original bounds
                if ub > value(node.ub):
                    ub = value(node.ub)  # don't make the bounds worse than the original bounds
                self.bnds_dict[node] = (lb, ub)

            if self.update_var_bounds:
                lb, ub = self.bnds_dict[node]
                if lb != -math.inf:
                    node.setlb(lb)
                if ub != math.inf:
                    node.setub(ub)
            return True, None

        if not node.is_expression_type():
            return True, None

        if node.__class__ in _prop_bnds_root_to_leaf_map:
            _prop_bnds_root_to_leaf_map[node.__class__](node, self.bnds_dict)
        else:
            logger.warning('Unsupported expression type for FBBT: {0}. Bounds will not be improved in this part of '
                           'the tree.'
                           ''.format(str(type(node))))

        return False, None


def fbbt_con(con, deactivate_satisfied_constraints=False, update_variable_bounds=True, integer_tol=1e-4):
    """
    Feasibility based bounds tightening for a constraint. This function attempts to improve the bounds of each variable
    in the constraint based on the bounds of the constraint and the bounds of the other variables in the constraint.
    For example:

    >>> import pyomo.environ as pe
    >>> from pyomo.contrib.fbbt.fbbt import fbbt
    >>> m = pe.ConcreteModel()
    >>> m.x = pe.Var(bounds=(-1,1))
    >>> m.y = pe.Var(bounds=(-2,2))
    >>> m.z = pe.Var()
    >>> m.c = pe.Constraint(expr=m.x*m.y + m.z == 1)
    >>> fbbt(m.c)
    >>> print(m.z.lb, m.z.ub)
    -1.0 3.0

    Parameters
    ----------
    con: pyomo.core.base.constraint.Constraint
        constraint on which to perform fbbt
    deactivate_satisfied_constraints: bool
        If deactivate_satisfied_constraints is True and the constraint is always satisfied, then the constranit
        will be deactivated
    update_variable_bounds: bool
        If update_variable_bounds is True, then the bounds on variables will be automatically updated.
    integer_tol: float
        If the lower bound computed on a binary variable is less than or equal to integer_tol, then the
        lower bound is left at 0. Otherwise, the lower bound is increased to 1. If the upper bound computed
        on a binary variable is greater than or equal to 1-integer_tol, then the upper bound is left at 1.
        Otherwise the upper bound is decreased to 0.

    Returns
    -------
    new_var_bounds: ComponentMap
        A ComponentMap mapping from variables a tuple containing the lower and upper bounds, respectively, computed
        from FBBT.
    """
    if not con.active:
        return None

    bnds_dict = ComponentMap()  # a dictionary to store the bounds of
    #  every node in the tree

    # a walker to propagate bounds from the variables to the root
    visitorA = _FBBTVisitorLeafToRoot(bnds_dict)
    visitorA.dfs_postorder_stack(con.body)

    # Now we need to replace the bounds in bnds_dict for the root
    # node with the bounds on the constraint (if those bounds are
    # better).
    _lb = value(con.lower)
    _ub = value(con.upper)
    if _lb is None:
        _lb = -math.inf
    if _ub is None:
        _ub = math.inf

    # first check if the constraint is always satisfied
    lb, ub = bnds_dict[con.body]
    if deactivate_satisfied_constraints:
        if lb >= _lb and ub <= _ub:
            con.deactivate()

    if _lb > lb:
        lb = _lb
    if _ub < ub:
        ub = _ub
    bnds_dict[con.body] = (lb, ub)

    # Now, propagate bounds back from the root to the variables
    visitorB = _FBBTVisitorRootToLeaf(bnds_dict, update_variable_bounds=update_variable_bounds, integer_tol=integer_tol)
    visitorB.dfs_postorder_stack(con.body)

    new_var_bounds = ComponentMap()
    for _node, _bnds in bnds_dict.items():
        if _node.__class__ in nonpyomo_leaf_types:
            continue
        if _node.is_variable_type():
            lb, ub = bnds_dict[_node]
            if lb == -math.inf:
                lb = None
            if ub == math.inf:
                ub = None
            new_var_bounds[_node] = (lb, ub)
    return new_var_bounds


def fbbt_block(m, tol=1e-4, deactivate_satisfied_constraints=False, update_variable_bounds=True, integer_tol=1e-4):
    """
    Feasibility based bounds tightening (FBBT) for a block or model. This
    loops through all of the constraints in the block and performs
    FBBT on each constraint (see the docstring for fbbt_con()).
    Through this processes, any variables whose bounds improve
    by more than tol are collected, and FBBT is
    performed again on all constraints involving those variables.
    This process is continued until no variable bounds are improved
    by more than tol.

    Parameters
    ----------
    m: pyomo.core.base.block.Block or pyomo.core.base.PyomoModel.ConcreteModel
    tol: float
    deactivate_satisfied_constraints: bool
        If deactivate_satisfied_constraints is True and a constraint is always satisfied, then the constranit
        will be deactivated
    update_variable_bounds: bool
        If update_variable_bounds is True, then the bounds on variables will be automatically updated.
    integer_tol: float
        If the lower bound computed on a binary variable is less than or equal to integer_tol, then the
        lower bound is left at 0. Otherwise, the lower bound is increased to 1. If the upper bound computed
        on a binary variable is greater than or equal to 1-integer_tol, then the upper bound is left at 1.
        Otherwise the upper bound is decreased to 0.

    Returns
    -------
    new_var_bounds: ComponentMap
        A ComponentMap mapping from variables a tuple containing the lower and upper bounds, respectively, computed
        from FBBT.
    """
    new_var_bounds = ComponentMap()
    var_to_con_map = ComponentMap()
    var_lbs = ComponentMap()
    var_ubs = ComponentMap()
    for c in m.component_data_objects(ctype=Constraint, active=True,
                                      descend_into=True, sort=True):
        for v in identify_variables(c.body):
            if v not in var_to_con_map:
                var_to_con_map[v] = list()
            if v.lb is None:
                var_lbs[v] = -math.inf
            else:
                var_lbs[v] = value(v.lb)
            if v.ub is None:
                var_ubs[v] = math.inf
            else:
                var_ubs[v] = value(v.ub)
            var_to_con_map[v].append(c)

    improved_vars = ComponentSet()
    for c in m.component_data_objects(ctype=Constraint, active=True,
                                      descend_into=True, sort=True):
        _new_var_bounds = fbbt_con(c, deactivate_satisfied_constraints=deactivate_satisfied_constraints,
                                   update_variable_bounds=update_variable_bounds, integer_tol=integer_tol)
        new_var_bounds.update(_new_var_bounds)
        for v in _new_var_bounds.keys():
            if v.lb is not None:
                if value(v.lb) > var_lbs[v] + tol:
                    improved_vars.add(v)
                    var_lbs[v] = value(v.lb)
            if v.ub is not None:
                if value(v.ub) < var_ubs[v] - tol:
                    improved_vars.add(v)
                    var_ubs[v] = value(v.ub)

    while len(improved_vars) > 0:
        v = improved_vars.pop()
        for c in var_to_con_map[v]:
            _new_var_bounds = fbbt_con(c, deactivate_satisfied_constraints=deactivate_satisfied_constraints,
                                       update_variable_bounds=update_variable_bounds, integer_tol=integer_tol)
            new_var_bounds.update(_new_var_bounds)
            for _v in _new_var_bounds.keys():
                if _v.lb is not None:
                    if value(_v.lb) > var_lbs[_v] + tol:
                        improved_vars.add(_v)
                        var_lbs[_v] = value(_v.lb)
                if _v.ub is not None:
                    if value(_v.ub) < var_ubs[_v] - tol:
                        improved_vars.add(_v)
                        var_ubs[_v] = value(_v.ub)

    return new_var_bounds


def fbbt(comp, deactivate_satisfied_constraints=False, update_variable_bounds=True, integer_tol=1e-4):
    """
    Perform FBBT on a constraint, block, or model. For more control,
    use fbbt_con and fbbt_block. For detailed documentation, see
    the docstrings for fbbt_con and fbbt_block.

    Parameters
    ----------
    comp: pyomo.core.base.constraint.Constraint or pyomo.core.base.block.Block or pyomo.core.base.PyomoModel.ConcreteModel
    deactivate_satisfied_constraints: bool
        If deactivate_satisfied_constraints is True and a constraint is always satisfied, then the constranit
        will be deactivated
    update_variable_bounds: bool
        If update_variable_bounds is True, then the bounds on variables will be automatically updated.
    integer_tol: float
        If the lower bound computed on a binary variable is less than or equal to integer_tol, then the
        lower bound is left at 0. Otherwise, the lower bound is increased to 1. If the upper bound computed
        on a binary variable is greater than or equal to 1-integer_tol, then the upper bound is left at 1.
        Otherwise the upper bound is decreased to 0.

    Returns
    -------
    new_var_bounds: ComponentMap
        A ComponentMap mapping from variables a tuple containing the lower and upper bounds, respectively, computed
        from FBBT.
    """
    new_var_bounds = ComponentMap()
    if comp.type() == Constraint:
        if comp.is_indexed():
            for _c in comp.values():
                _new_var_bounds = fbbt_con(comp, deactivate_satisfied_constraints=deactivate_satisfied_constraints,
                                           update_variable_bounds=update_variable_bounds, integer_tol=integer_tol)
                new_var_bounds.update(_new_var_bounds)
        else:
            _new_var_bounds = fbbt_con(comp, deactivate_satisfied_constraints=deactivate_satisfied_constraints,
                                       update_variable_bounds=update_variable_bounds, integer_tol=integer_tol)
            new_var_bounds.update(_new_var_bounds)
    elif comp.type() == Block:
        _new_var_bounds = fbbt_block(comp, deactivate_satisfied_constraints=deactivate_satisfied_constraints,
                                     update_variable_bounds=update_variable_bounds, integer_tol=integer_tol)
        new_var_bounds.update(_new_var_bounds)
    else:
        raise FBBTException('Cannot perform FBBT on objects of type {0}'.format(type(comp)))

    return new_var_bounds


def compute_bounds_on_expr(expr):
    """
    Compute bounds on an expression based on the bounds on the variables in the expression.

    Parameters
    ----------
    expr: pyomo.core.expr.expr_pyomo5.ExpressionBase

    Returns
    -------
    lb: float
    ub: float
    """
    bnds_dict = ComponentMap()
    visitor = _FBBTVisitorLeafToRoot(bnds_dict)
    visitor.dfs_postorder_stack(expr)

    return bnds_dict[expr]
