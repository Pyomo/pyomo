#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math


"""
The purpose of this file is to perform symbolic differentiation and 
first order automatic differentiation directly with pyomo 
expressions. This is certainly not as efficient as doing AD in C or 
C++, but it avoids the translation from pyomo expressions to a form 
where AD can be performed efficiently. The only functions that are 
meant to be used by users are reverse_ad and reverse_sd. First, 
values are propagated from the leaves to each node in the tree with 
the LeafToRoot visitors. Then derivative values are propagated from 
the root to the leaves with the RootToLeaf visitors.
"""


class DifferentiationException(Exception):
    pass


def _diff_ProductExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    der = der_dict[node]
    der_dict[arg1] += der * val_dict[arg2]
    der_dict[arg2] += der * val_dict[arg1]


def _diff_SumExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.SumExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    der = der_dict[node]
    for arg in node.args:
        der_dict[arg] += der


def _diff_PowExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.PowExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    der = der_dict[node]
    val1 = val_dict[arg1]
    val2 = val_dict[arg2]
    der_dict[arg1] += der * val2 * val1 ** (val2 - 1)
    if arg2.__class__ not in nonpyomo_leaf_types:
        der_dict[arg2] += der * val1**val2 * log(val1)


def _diff_DivisionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.DivisionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 2
    num = node.args[0]
    den = node.args[1]
    der = der_dict[node]
    der_dict[num] += der * (1 / val_dict[den])
    der_dict[den] -= der * val_dict[num] / val_dict[den] ** 2


def _diff_NegationExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] -= der


def _diff_exp(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der * exp(val_dict[arg])


def _diff_log(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der / val_dict[arg]


def _diff_log10(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der * math.log10(math.exp(1)) / val_dict[arg]


def _diff_sin(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der * cos(val_dict[arg])


def _diff_cos(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] -= der * sin(val_dict[arg])


def _diff_tan(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der / (cos(val_dict[arg]) ** 2)


def _diff_asin(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der / (1 - val_dict[arg] ** 2) ** 0.5


def _diff_acos(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] -= der / (1 - val_dict[arg] ** 2) ** 0.5


def _diff_atan(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der / (1 + val_dict[arg] ** 2)


def _diff_sqrt(node, val_dict, der_dict):
    """
    Reverse automatic differentiation on the square root function.
    Implementation copied from power function, with fixed exponent.

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der * 0.5 * val_dict[arg] ** (-0.5)


def _diff_abs(node, val_dict, der_dict):
    """
    Reverse automatic differentiation on the abs function.
    This will raise an exception at 0.

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    val = val_dict[arg]
    if is_constant(val) and val == 0:
        raise DifferentiationException('Cannot differentiate abs(x) at x=0')
    der_dict[arg] += der * val / abs(val)


_unary_map = dict()
_unary_map['exp'] = _diff_exp
_unary_map['log'] = _diff_log
_unary_map['log10'] = _diff_log10
_unary_map['sin'] = _diff_sin
_unary_map['cos'] = _diff_cos
_unary_map['tan'] = _diff_tan
_unary_map['asin'] = _diff_asin
_unary_map['acos'] = _diff_acos
_unary_map['atan'] = _diff_atan
_unary_map['sqrt'] = _diff_sqrt
_unary_map['abs'] = _diff_abs


def _diff_UnaryFunctionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    if node.getname() in _unary_map:
        _unary_map[node.getname()](node, val_dict, der_dict)
    else:
        raise DifferentiationException(
            'Unsupported expression type for differentiation: {0}'.format(type(node))
        )


def _diff_GeneralExpression(node, val_dict, der_dict):
    """
    Reverse automatic differentiation for named expressions.

    Parameters
    ----------
    node: The named expression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    der_dict[node.arg(0)] += der_dict[node]


def _diff_ExternalFunctionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ExternalFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    der = der_dict[node]
    vals = tuple(val_dict[i] for i in node.args)
    derivs = node._fcn.evaluate_fgh(vals, fgh=1)[1]
    for ndx, arg in enumerate(node.args):
        der_dict[arg] += der * derivs[ndx]


_diff_map = dict()
_diff_map[_expr.ProductExpression] = _diff_ProductExpression
_diff_map[_expr.DivisionExpression] = _diff_DivisionExpression
_diff_map[_expr.PowExpression] = _diff_PowExpression
_diff_map[_expr.SumExpression] = _diff_SumExpression
_diff_map[_expr.MonomialTermExpression] = _diff_ProductExpression
_diff_map[_expr.NegationExpression] = _diff_NegationExpression
_diff_map[_expr.UnaryFunctionExpression] = _diff_UnaryFunctionExpression
_diff_map[_expr.ExternalFunctionExpression] = _diff_ExternalFunctionExpression
_diff_map[_expr.LinearExpression] = _diff_SumExpression
_diff_map[_expr.AbsExpression] = _diff_abs

_diff_map[_expr.NPV_ProductExpression] = _diff_ProductExpression
_diff_map[_expr.NPV_DivisionExpression] = _diff_DivisionExpression
_diff_map[_expr.NPV_PowExpression] = _diff_PowExpression
_diff_map[_expr.NPV_SumExpression] = _diff_SumExpression
_diff_map[_expr.NPV_NegationExpression] = _diff_NegationExpression
_diff_map[_expr.NPV_UnaryFunctionExpression] = _diff_UnaryFunctionExpression
_diff_map[_expr.NPV_ExternalFunctionExpression] = _diff_ExternalFunctionExpression
_diff_map[_expr.NPV_AbsExpression] = _diff_abs


def _symbolic_value(x):
    return x


def _numeric_apply_operation(node, values):
    return node._apply_operation(values)


def _symbolic_apply_operation(node, values):
    return node


class _LeafToRootVisitor(ExpressionValueVisitor):
    def __init__(self, val_dict, der_dict, expr_list, numeric=True):
        """
        Parameters
        ----------
        val_dict: ComponentMap
        der_dict: ComponentMap
        """
        self.val_dict = val_dict
        self.der_dict = der_dict
        self.expr_list = expr_list
        assert len(self.expr_list) == 0
        assert len(self.val_dict) == 0
        assert len(self.der_dict) == 0
        if numeric:
            self.value_func = value
            self.operation_func = _numeric_apply_operation
        else:
            self.value_func = _symbolic_value
            self.operation_func = _symbolic_apply_operation

    def visit(self, node, values):
        self.val_dict[node] = self.operation_func(node, values)
        self.der_dict[node] = 0
        self.expr_list.append(node)
        return self.val_dict[node]

    def visiting_potential_leaf(self, node):
        if node in self.val_dict:
            return True, self.val_dict[node]

        if node.__class__ in nonpyomo_leaf_types:
            self.val_dict[node] = node
            self.der_dict[node] = 0
            return True, node

        if not node.is_expression_type():
            val = self.value_func(node)
            self.val_dict[node] = val
            self.der_dict[node] = 0
            return True, val

        return False, None


def _reverse_diff_helper(expr, numeric=True):
    val_dict = ComponentMap()
    der_dict = ComponentMap()
    expr_list = list()

    visitorA = _LeafToRootVisitor(val_dict, der_dict, expr_list, numeric=numeric)
    visitorA.dfs_postorder_stack(expr)

    der_dict[expr] = 1
    for e in reversed(expr_list):
        if e.__class__ in _diff_map:
            _diff_map[e.__class__](e, val_dict, der_dict)
        elif e.is_named_expression_type():
            _diff_GeneralExpression(e, val_dict, der_dict)
        else:
            raise DifferentiationException(
                'Unsupported expression type for differentiation: {0}'.format(type(e))
            )

    return der_dict


def reverse_ad(expr):
    """
    First order reverse automatic differentiation

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.NumericExpression
        expression to differentiate

    Returns
    -------
    ComponentMap
        component_map mapping variables to derivatives with respect
        to the corresponding variable
    """
    return _reverse_diff_helper(expr, True)


def reverse_sd(expr):
    """
    First order reverse symbolic differentiation

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.NumericExpression
        expression to differentiate

    Returns
    -------
    ComponentMap
        component_map mapping variables to derivatives with respect
        to the corresponding variable
    """
    return _reverse_diff_helper(expr, False)
