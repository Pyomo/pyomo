#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, identify_variables
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, value
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, SimpleExpression
import logging
from pyomo.common.errors import InfeasibleConstraintException, PyomoException
from pyomo.common.config import ConfigBlock, ConfigValue, In, NonNegativeFloat, NonNegativeInt

logger = logging.getLogger(__name__)


"""
The purpose of this file is to perform feasibility based bounds 
tightening. This is a very basic implementation, but it is done 
directly with pyomo expressions. The only functions that are meant to 
be used by users are fbbt and compute_bounds_on_expr. The first set of 
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


class FBBTException(PyomoException):
    pass


def _prop_bnds_leaf_to_root_ProductExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    bnds_dict[node] = interval.mul(lb1, ub1, lb2, ub2)


def _prop_bnds_leaf_to_root_SumExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.SumExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    arg0 = node.arg(0)
    lb, ub = bnds_dict[arg0]
    for i in range(1, node.nargs()):
        arg = node.arg(i)
        lb2, ub2 = bnds_dict[arg]
        lb, ub = interval.add(lb, ub, lb2, ub2)
    bnds_dict[node] = (lb, ub)


def _prop_bnds_leaf_to_root_DivisionExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.DivisionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    bnds_dict[node] = interval.div(lb1, ub1, lb2, ub2, feasibility_tol=feasibility_tol)


def _prop_bnds_leaf_to_root_PowExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.PowExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    bnds_dict[node] = interval.power(lb1, ub1, lb2, ub2)


def _prop_bnds_leaf_to_root_ReciprocalExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ReciprocalExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.inv(lb1, ub1, feasibility_tol)


def _prop_bnds_leaf_to_root_NegationExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.sub(0, 0, lb1, ub1)


def _prop_bnds_leaf_to_root_exp(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.exp(lb1, ub1)


def _prop_bnds_leaf_to_root_log(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.log(lb1, ub1)


def _prop_bnds_leaf_to_root_log10(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.log10(lb1, ub1)


def _prop_bnds_leaf_to_root_sin(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.sin(lb1, ub1)


def _prop_bnds_leaf_to_root_cos(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.cos(lb1, ub1)


def _prop_bnds_leaf_to_root_tan(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.tan(lb1, ub1)


def _prop_bnds_leaf_to_root_asin(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.asin(lb1, ub1, -interval.inf, interval.inf, feasibility_tol)


def _prop_bnds_leaf_to_root_acos(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.acos(lb1, ub1, -interval.inf, interval.inf, feasibility_tol)


def _prop_bnds_leaf_to_root_atan(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.atan(lb1, ub1, -interval.inf, interval.inf)


def _prop_bnds_leaf_to_root_sqrt(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb1, ub1 = bnds_dict[arg]
    bnds_dict[node] = interval.power(lb1, ub1, 0.5, 0.5)


_unary_leaf_to_root_map = dict()
_unary_leaf_to_root_map['exp'] = _prop_bnds_leaf_to_root_exp
_unary_leaf_to_root_map['log'] = _prop_bnds_leaf_to_root_log
_unary_leaf_to_root_map['log10'] = _prop_bnds_leaf_to_root_log10
_unary_leaf_to_root_map['sin'] = _prop_bnds_leaf_to_root_sin
_unary_leaf_to_root_map['cos'] = _prop_bnds_leaf_to_root_cos
_unary_leaf_to_root_map['tan'] = _prop_bnds_leaf_to_root_tan
_unary_leaf_to_root_map['asin'] = _prop_bnds_leaf_to_root_asin
_unary_leaf_to_root_map['acos'] = _prop_bnds_leaf_to_root_acos
_unary_leaf_to_root_map['atan'] = _prop_bnds_leaf_to_root_atan
_unary_leaf_to_root_map['sqrt'] = _prop_bnds_leaf_to_root_sqrt


def _prop_bnds_leaf_to_root_UnaryFunctionExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    if node.getname() in _unary_leaf_to_root_map:
        _unary_leaf_to_root_map[node.getname()](node, bnds_dict, feasibility_tol)
    else:
        bnds_dict[node] = (-interval.inf, interval.inf)


def _prop_bnds_leaf_to_root_GeneralExpression(node, bnds_dict, feasibility_tol):
    """
    Propagate bounds from children to parent

    Parameters
    ----------
    node: pyomo.core.base.expression._GeneralExpressionData
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    expr_lb, expr_ub = bnds_dict[node.expr]
    bnds_dict[node] = (expr_lb, expr_ub)


_prop_bnds_leaf_to_root_map = dict()
_prop_bnds_leaf_to_root_map[numeric_expr.ProductExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[numeric_expr.DivisionExpression] = _prop_bnds_leaf_to_root_DivisionExpression
_prop_bnds_leaf_to_root_map[numeric_expr.ReciprocalExpression] = _prop_bnds_leaf_to_root_ReciprocalExpression
_prop_bnds_leaf_to_root_map[numeric_expr.PowExpression] = _prop_bnds_leaf_to_root_PowExpression
_prop_bnds_leaf_to_root_map[numeric_expr.SumExpression] = _prop_bnds_leaf_to_root_SumExpression
_prop_bnds_leaf_to_root_map[numeric_expr.MonomialTermExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[numeric_expr.NegationExpression] = _prop_bnds_leaf_to_root_NegationExpression
_prop_bnds_leaf_to_root_map[numeric_expr.UnaryFunctionExpression] = _prop_bnds_leaf_to_root_UnaryFunctionExpression

_prop_bnds_leaf_to_root_map[numeric_expr.NPV_ProductExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[numeric_expr.NPV_DivisionExpression] = _prop_bnds_leaf_to_root_DivisionExpression
_prop_bnds_leaf_to_root_map[numeric_expr.NPV_ReciprocalExpression] = _prop_bnds_leaf_to_root_ReciprocalExpression
_prop_bnds_leaf_to_root_map[numeric_expr.NPV_PowExpression] = _prop_bnds_leaf_to_root_PowExpression
_prop_bnds_leaf_to_root_map[numeric_expr.NPV_SumExpression] = _prop_bnds_leaf_to_root_SumExpression
_prop_bnds_leaf_to_root_map[numeric_expr.NPV_NegationExpression] = _prop_bnds_leaf_to_root_NegationExpression
_prop_bnds_leaf_to_root_map[numeric_expr.NPV_UnaryFunctionExpression] = _prop_bnds_leaf_to_root_UnaryFunctionExpression

_prop_bnds_leaf_to_root_map[_GeneralExpressionData] = _prop_bnds_leaf_to_root_GeneralExpression
_prop_bnds_leaf_to_root_map[SimpleExpression] = _prop_bnds_leaf_to_root_GeneralExpression


def _prop_bnds_root_to_leaf_ProductExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    _lb1, _ub1 = interval.div(lb0, ub0, lb2, ub2, feasibility_tol)
    _lb2, _ub2 = interval.div(lb0, ub0, lb1, ub1, feasibility_tol)
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


def _prop_bnds_root_to_leaf_SumExpression(node, bnds_dict, feasibility_tol):
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
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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


def _prop_bnds_root_to_leaf_DivisionExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.DivisionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    _lb1, _ub1 = interval.mul(lb0, ub0, lb2, ub2)
    _lb2, _ub2 = interval.div(lb1, ub1, lb0, ub0, feasibility_tol=feasibility_tol)
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


def _prop_bnds_root_to_leaf_PowExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.PowExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = bnds_dict[arg2]
    _lb1, _ub1 = interval._inverse_power1(lb0, ub0, lb2, ub2, orig_xl=lb1, orig_xu=ub1, feasibility_tol=feasibility_tol)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg1] = (lb1, ub1)

    if is_fixed(arg2) and lb2 == ub2:  # No need to tighten the bounds on arg2 if arg2 is fixed
        pass
    else:
        _lb2, _ub2 = interval._inverse_power2(lb0, ub0, lb1, ub1, feasiblity_tol=feasibility_tol)
        if _lb2 > lb2:
            lb2 = _lb2
        if _ub2 < ub2:
            ub2 = _ub2
        bnds_dict[arg2] = (lb2, ub2)


def _prop_bnds_root_to_leaf_sqrt(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg1 = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg1]
    lb2, ub2 = (0.5, 0.5)
    _lb1, _ub1 = interval._inverse_power1(lb0, ub0, lb2, ub2, orig_xl=lb1, orig_xu=ub1, feasibility_tol=feasibility_tol)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg1] = (lb1, ub1)


def _prop_bnds_root_to_leaf_ReciprocalExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ReciprocalExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.inv(lb0, ub0, feasibility_tol)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_NegationExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.NegationExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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


def _prop_bnds_root_to_leaf_exp(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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


def _prop_bnds_root_to_leaf_log(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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


def _prop_bnds_root_to_leaf_log10(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.power(10, 10, lb0, ub0)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_sin(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.asin(lb0, ub0, lb1, ub1, feasibility_tol)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_cos(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    assert len(node.args) == 1
    arg = node.args[0]
    lb0, ub0 = bnds_dict[node]
    lb1, ub1 = bnds_dict[arg]
    _lb1, _ub1 = interval.acos(lb0, ub0, lb1, ub1, feasibility_tol)
    if _lb1 > lb1:
        lb1 = _lb1
    if _ub1 < ub1:
        ub1 = _ub1
    bnds_dict[arg] = (lb1, ub1)


def _prop_bnds_root_to_leaf_tan(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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


def _prop_bnds_root_to_leaf_asin(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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


def _prop_bnds_root_to_leaf_acos(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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


def _prop_bnds_root_to_leaf_atan(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
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
_unary_root_to_leaf_map['log10'] = _prop_bnds_root_to_leaf_log10
_unary_root_to_leaf_map['sin'] = _prop_bnds_root_to_leaf_sin
_unary_root_to_leaf_map['cos'] = _prop_bnds_root_to_leaf_cos
_unary_root_to_leaf_map['tan'] = _prop_bnds_root_to_leaf_tan
_unary_root_to_leaf_map['asin'] = _prop_bnds_root_to_leaf_asin
_unary_root_to_leaf_map['acos'] = _prop_bnds_root_to_leaf_acos
_unary_root_to_leaf_map['atan'] = _prop_bnds_root_to_leaf_atan
_unary_root_to_leaf_map['sqrt'] = _prop_bnds_root_to_leaf_sqrt


def _prop_bnds_root_to_leaf_UnaryFunctionExpression(node, bnds_dict, feasibility_tol):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    if node.getname() in _unary_root_to_leaf_map:
        _unary_root_to_leaf_map[node.getname()](node, bnds_dict, feasibility_tol)
    else:
        logger.warning('Unsupported expression type for FBBT: {0}. Bounds will not be improved in this part of '
                       'the tree.'
                       ''.format(node.getname()))


def _prop_bnds_root_to_leaf_GeneralExpression(node, bnds_dict, feasibility_tol):
    """
    Propagate bounds from parent to children.

    Parameters
    ----------
    node: pyomo.core.base.expression._GeneralExpressionData
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    expr_lb, expr_ub = bnds_dict[node]
    bnds_dict[node.expr] = (expr_lb, expr_ub)


_prop_bnds_root_to_leaf_map = dict()
_prop_bnds_root_to_leaf_map[numeric_expr.ProductExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[numeric_expr.DivisionExpression] = _prop_bnds_root_to_leaf_DivisionExpression
_prop_bnds_root_to_leaf_map[numeric_expr.ReciprocalExpression] = _prop_bnds_root_to_leaf_ReciprocalExpression
_prop_bnds_root_to_leaf_map[numeric_expr.PowExpression] = _prop_bnds_root_to_leaf_PowExpression
_prop_bnds_root_to_leaf_map[numeric_expr.SumExpression] = _prop_bnds_root_to_leaf_SumExpression
_prop_bnds_root_to_leaf_map[numeric_expr.MonomialTermExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[numeric_expr.NegationExpression] = _prop_bnds_root_to_leaf_NegationExpression
_prop_bnds_root_to_leaf_map[numeric_expr.UnaryFunctionExpression] = _prop_bnds_root_to_leaf_UnaryFunctionExpression

_prop_bnds_root_to_leaf_map[numeric_expr.NPV_ProductExpression] = _prop_bnds_root_to_leaf_ProductExpression
_prop_bnds_root_to_leaf_map[numeric_expr.NPV_DivisionExpression] = _prop_bnds_root_to_leaf_DivisionExpression
_prop_bnds_root_to_leaf_map[numeric_expr.NPV_ReciprocalExpression] = _prop_bnds_root_to_leaf_ReciprocalExpression
_prop_bnds_root_to_leaf_map[numeric_expr.NPV_PowExpression] = _prop_bnds_root_to_leaf_PowExpression
_prop_bnds_root_to_leaf_map[numeric_expr.NPV_SumExpression] = _prop_bnds_root_to_leaf_SumExpression
_prop_bnds_root_to_leaf_map[numeric_expr.NPV_NegationExpression] = _prop_bnds_root_to_leaf_NegationExpression
_prop_bnds_root_to_leaf_map[numeric_expr.NPV_UnaryFunctionExpression] = _prop_bnds_root_to_leaf_UnaryFunctionExpression

_prop_bnds_root_to_leaf_map[_GeneralExpressionData] = _prop_bnds_root_to_leaf_GeneralExpression
_prop_bnds_root_to_leaf_map[SimpleExpression] = _prop_bnds_root_to_leaf_GeneralExpression


def _check_and_reset_bounds(var, lb, ub):
    """
    This function ensures that lb is not less than var.lb and that ub is not greater than var.ub.
    """
    orig_lb = value(var.lb)
    orig_ub = value(var.ub)
    if orig_lb is None:
        orig_lb = -interval.inf
    if orig_ub is None:
        orig_ub = interval.inf
    if lb < orig_lb:
        lb = orig_lb
    if ub > orig_ub:
        ub = orig_ub
    return lb, ub


class _FBBTVisitorLeafToRoot(ExpressionValueVisitor):
    """
    This walker propagates bounds from the variables to each node in
    the expression tree (all the way to the root node).
    """
    def __init__(self, bnds_dict, integer_tol=1e-4, feasibility_tol=1e-8):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        integer_tol: float
        feasibility_tol: float
            If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
            feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
            is also used when performing certain interval arithmetic operations to ensure that none of the feasible
            region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
            is more conservative).
        """
        self.bnds_dict = bnds_dict
        self.integer_tol = integer_tol
        self.feasibility_tol = feasibility_tol

    def visit(self, node, values):
        if node.__class__ in _prop_bnds_leaf_to_root_map:
            _prop_bnds_leaf_to_root_map[node.__class__](node, self.bnds_dict, self.feasibility_tol)
        else:
            self.bnds_dict[node] = (-interval.inf, interval.inf)
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
                    lb = -interval.inf
                if ub is None:
                    ub = interval.inf
                if lb - self.feasibility_tol > ub:
                    raise InfeasibleConstraintException('Variable has a lower bound which is larger than its upper bound: {0}'.format(str(node)))
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
    def __init__(self, bnds_dict, integer_tol=1e-4, feasibility_tol=1e-8):
        """
        Parameters
        ----------
        bnds_dict: ComponentMap
        integer_tol: float
        feasibility_tol: float
            If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
            feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
            is also used when performing certain interval arithmetic operations to ensure that none of the feasible
            region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
            is more conservative).
        """
        self.bnds_dict = bnds_dict
        self.integer_tol = integer_tol
        self.feasibility_tol = feasibility_tol

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            lb, ub = self.bnds_dict[node]
            if abs(lb - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            if abs(ub - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            return True, None

        if node.is_variable_type():
            lb, ub = self.bnds_dict[node]

            lb, ub = self.bnds_dict[node]
            if lb > ub:
                if lb - self.feasibility_tol > ub:
                    raise InfeasibleConstraintException('Lower bound ({1}) computed for variable {0} is larger than the computed upper bound ({2}).'.format(node, lb, ub))
                else:
                    """
                    If we reach this code, then lb > ub, but not by more than feasibility_tol. 
                    Now we want to decrease lb slightly and increase ub slightly so that lb <= ub.
                    However, we also have to make sure we do not make lb lower than the original lower bound
                    and make sure we do not make ub larger than the original upper bound. This is what 
                    _check_and_reset_bounds is for.
                    """
                    lb -= self.feasibility_tol
                    ub += self.feasibility_tol
                    lb, ub = _check_and_reset_bounds(node, lb, ub)
                    self.bnds_dict[node] = (lb, ub)
            if lb == interval.inf:
                raise InfeasibleConstraintException('Computed a lower bound of +inf for variable {0}'.format(node))
            if ub == -interval.inf:
                raise InfeasibleConstraintException('Computed an upper bound of -inf for variable {0}'.format(node))

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
                if lb > -interval.inf:
                    lb = max(math.floor(lb), math.ceil(lb - self.integer_tol))
                if ub < interval.inf:
                    ub = min(math.ceil(ub), math.floor(ub + self.integer_tol))
                """
                We have to make sure we do not make lb lower than the original lower bound
                and make sure we do not make ub larger than the original upper bound. This is what 
                _check_and_reset_bounds is for.
                """
                lb, ub = _check_and_reset_bounds(node, lb, ub)
                self.bnds_dict[node] = (lb, ub)

            if lb != -interval.inf:
                node.setlb(lb)
            if ub != interval.inf:
                node.setub(ub)
            return True, None

        if not node.is_expression_type():
            lb, ub = self.bnds_dict[node]
            if abs(lb - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            if abs(ub - value(node)) > self.feasibility_tol:
                raise InfeasibleConstraintException('Detected an infeasible constraint.')
            return True, None

        if node.__class__ in _prop_bnds_root_to_leaf_map:
            _prop_bnds_root_to_leaf_map[node.__class__](node, self.bnds_dict, self.feasibility_tol)
        else:
            logger.warning('Unsupported expression type for FBBT: {0}. Bounds will not be improved in this part of '
                           'the tree.'
                           ''.format(str(type(node))))

        return False, None


def _fbbt_con(con, config):
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
    config: ConfigBlock
        see documentation for fbbt

    Returns
    -------
    new_var_bounds: ComponentMap
        A ComponentMap mapping from variables a tuple containing the lower and upper bounds, respectively, computed
        from FBBT.
    """
    if not con.active:
        return ComponentMap()

    bnds_dict = ComponentMap()  # a dictionary to store the bounds of every node in the tree

    # a walker to propagate bounds from the variables to the root
    visitorA = _FBBTVisitorLeafToRoot(bnds_dict, feasibility_tol=config.feasibility_tol)
    visitorA.dfs_postorder_stack(con.body)

    # Now we need to replace the bounds in bnds_dict for the root
    # node with the bounds on the constraint (if those bounds are
    # better).
    _lb = value(con.lower)
    _ub = value(con.upper)
    if _lb is None:
        _lb = -interval.inf
    if _ub is None:
        _ub = interval.inf

    lb, ub = bnds_dict[con.body]

    # check if the constraint is infeasible
    if lb > _ub + config.feasibility_tol or ub < _lb - config.feasibility_tol:
        raise InfeasibleConstraintException('Detected an infeasible constraint during FBBT: {0}'.format(str(con)))

    # check if the constraint is always satisfied
    if config.deactivate_satisfied_constraints:
        if lb >= _lb - config.feasibility_tol and ub <= _ub + config.feasibility_tol:
            con.deactivate()

    if _lb > lb:
        lb = _lb
    if _ub < ub:
        ub = _ub
    bnds_dict[con.body] = (lb, ub)

    # Now, propagate bounds back from the root to the variables
    visitorB = _FBBTVisitorRootToLeaf(bnds_dict, integer_tol=config.integer_tol, feasibility_tol=config.feasibility_tol)
    visitorB.dfs_postorder_stack(con.body)

    new_var_bounds = ComponentMap()
    for _node, _bnds in bnds_dict.items():
        if _node.__class__ in nonpyomo_leaf_types:
            continue
        if _node.is_variable_type():
            lb, ub = bnds_dict[_node]
            if lb == -interval.inf:
                lb = None
            if ub == interval.inf:
                ub = None
            new_var_bounds[_node] = (lb, ub)
    return new_var_bounds


def _fbbt_block(m, config):
    """
    Feasibility based bounds tightening (FBBT) for a block or model. This
    loops through all of the constraints in the block and performs
    FBBT on each constraint (see the docstring for _fbbt_con()).
    Through this processes, any variables whose bounds improve
    by more than tol are collected, and FBBT is
    performed again on all constraints involving those variables.
    This process is continued until no variable bounds are improved
    by more than tol.

    Parameters
    ----------
    m: pyomo.core.base.block.Block or pyomo.core.base.PyomoModel.ConcreteModel
    config: ConfigBlock
        See the docs for fbbt

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
    n_cons = 0
    for c in m.component_data_objects(ctype=Constraint, active=True,
                                      descend_into=True, sort=True):
        for v in identify_variables(c.body):
            if v not in var_to_con_map:
                var_to_con_map[v] = list()
            if v.lb is None:
                var_lbs[v] = -interval.inf
            else:
                var_lbs[v] = value(v.lb)
            if v.ub is None:
                var_ubs[v] = interval.inf
            else:
                var_ubs[v] = value(v.ub)
            var_to_con_map[v].append(c)
        n_cons += 1

    for _v in m.component_data_objects(ctype=Var, active=True, descend_into=True, sort=True):
        if _v.is_fixed():
            _v.setlb(_v.value)
            _v.setub(_v.value)
            new_var_bounds[_v] = (_v.value, _v.value)

    n_fbbt = 0

    improved_vars = ComponentSet()
    for c in m.component_data_objects(ctype=Constraint, active=True,
                                      descend_into=True, sort=True):
        _new_var_bounds = _fbbt_con(c, config)
        n_fbbt += 1
        new_var_bounds.update(_new_var_bounds)
        for v, bnds in _new_var_bounds.items():
            vlb, vub = bnds
            if vlb is not None:
                if vlb > var_lbs[v] + config.improvement_tol:
                    improved_vars.add(v)
                    var_lbs[v] = vlb
            if vub is not None:
                if vub < var_ubs[v] - config.improvement_tol:
                    improved_vars.add(v)
                    var_ubs[v] = vub

    while len(improved_vars) > 0:
        if n_fbbt >= n_cons * config.max_iter:
            break
        v = improved_vars.pop()
        for c in var_to_con_map[v]:
            _new_var_bounds = _fbbt_con(c, config)
            n_fbbt += 1
            new_var_bounds.update(_new_var_bounds)
            for _v, bnds in _new_var_bounds.items():
                _vlb, _vub = bnds
                if _vlb is not None:
                    if _vlb > var_lbs[_v] + config.improvement_tol:
                        improved_vars.add(_v)
                        var_lbs[_v] = _vlb
                if _vub is not None:
                    if _vub < var_ubs[_v] - config.improvement_tol:
                        improved_vars.add(_v)
                        var_ubs[_v] = _vub

    return new_var_bounds


def fbbt(comp, deactivate_satisfied_constraints=False, integer_tol=1e-5, feasibility_tol=1e-8, max_iter=10,
         improvement_tol=1e-4):
    """
    Perform FBBT on a constraint, block, or model. For more control,
    use _fbbt_con and _fbbt_block. For detailed documentation, see
    the docstrings for _fbbt_con and _fbbt_block.

    Parameters
    ----------
    comp: pyomo.core.base.constraint.Constraint or pyomo.core.base.block.Block or pyomo.core.base.PyomoModel.ConcreteModel
    deactivate_satisfied_constraints: bool
        If deactivate_satisfied_constraints is True and a constraint is always satisfied, then the constranit
        will be deactivated
    integer_tol: float
        If the lower bound computed on a binary variable is less than or equal to integer_tol, then the
        lower bound is left at 0. Otherwise, the lower bound is increased to 1. If the upper bound computed
        on a binary variable is greater than or equal to 1-integer_tol, then the upper bound is left at 1.
        Otherwise the upper bound is decreased to 0.
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    max_iter: int
        Used for Blocks only (i.e., comp.ctype == Block). When performing FBBT on a Block, we first perform FBBT on
        every constraint in the Block. We then attempt to identify which constraints to repeat FBBT on based on the
        improvement in variable bounds. If the bounds on a variable improve by more than improvement_tol, then FBBT
        is performed on the constraints using that Var. However, this algorithm is not guaranteed to converge, so
        max_iter limits the total number of times FBBT is performed to max_iter times the number of constraints
        in the Block.
    improvement_tol: float
        Used for Blocks only (i.e., comp.ctype == Block). When performing FBBT on a Block, we first perform FBBT on
        every constraint in the Block. We then attempt to identify which constraints to repeat FBBT on based on the
        improvement in variable bounds. If the bounds on a variable improve by more than improvement_tol, then FBBT
        is performed on the constraints using that Var.

    Returns
    -------
    new_var_bounds: ComponentMap
        A ComponentMap mapping from variables a tuple containing the lower and upper bounds, respectively, computed
        from FBBT.
    """
    config = ConfigBlock()
    dsc_config = ConfigValue(default=deactivate_satisfied_constraints, domain=In({True, False}))
    integer_tol_config = ConfigValue(default=integer_tol, domain=NonNegativeFloat)
    ft_config = ConfigValue(default=feasibility_tol, domain=NonNegativeFloat)
    mi_config = ConfigValue(default=max_iter, domain=NonNegativeInt)
    improvement_tol_config = ConfigValue(default=improvement_tol, domain=NonNegativeFloat)
    config.declare('deactivate_satisfied_constraints', dsc_config)
    config.declare('integer_tol', integer_tol_config)
    config.declare('feasibility_tol', ft_config)
    config.declare('max_iter', mi_config)
    config.declare('improvement_tol', improvement_tol_config)

    new_var_bounds = ComponentMap()
    if comp.ctype == Constraint:
        if comp.is_indexed():
            for _c in comp.values():
                _new_var_bounds = _fbbt_con(comp, config)
                new_var_bounds.update(_new_var_bounds)
        else:
            _new_var_bounds = _fbbt_con(comp, config)
            new_var_bounds.update(_new_var_bounds)
    elif comp.ctype in {Block, Disjunct}:
        _new_var_bounds = _fbbt_block(comp, config)
        new_var_bounds.update(_new_var_bounds)
    else:
        raise FBBTException('Cannot perform FBBT on objects of type {0}'.format(type(comp)))

    return new_var_bounds


def compute_bounds_on_expr(expr):
    """
    Compute bounds on an expression based on the bounds on the variables in the expression.

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.ExpressionBase

    Returns
    -------
    lb: float
    ub: float
    """
    bnds_dict = ComponentMap()
    visitor = _FBBTVisitorLeafToRoot(bnds_dict)
    visitor.dfs_postorder_stack(expr)
    lb, ub = bnds_dict[expr]
    if lb == -interval.inf:
        lb = None
    if ub == interval.inf:
        ub = None

    return lb, ub


class BoundsManager(object):
    def __init__(self, comp):
        self._vars = ComponentSet()
        self._saved_bounds = list()

        if comp.ctype == Constraint:
            if comp.is_indexed():
                for c in comp.values():
                    self._vars.update(identify_variables(c.body))
            else:
                self._vars.update(identify_variables(comp.body))
        else:
            for c in comp.component_data_objects(Constraint, descend_into=True, active=True, sort=True):
                self._vars.update(identify_variables(c.body))

    def save_bounds(self):
        bnds = ComponentMap()
        for v in self._vars:
            bnds[v] = (v.lb, v.ub)
        self._saved_bounds.append(bnds)

    def pop_bounds(self, ndx=-1):
        bnds = self._saved_bounds.pop(ndx)
        for v, _bnds in bnds.items():
            lb, ub = _bnds
            v.setlb(lb)
            v.setub(ub)

    def load_bounds(self, bnds, save_current_bounds=True):
        if save_current_bounds:
            self.save_bounds()
        for v, _bnds in bnds.items():
            if v in self._vars:
                lb, ub = _bnds
                v.setlb(lb)
                v.setub(ub)
