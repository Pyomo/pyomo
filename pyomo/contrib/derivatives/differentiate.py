from pyomo.core.kernel.component_map import ComponentMap
import pyomo.core.expr.expr_pyomo5 as _expr
from pyomo.core.expr.expr_pyomo5 import ExpressionValueVisitor, nonpyomo_leaf_types, value
from pyomo.core.expr.current import exp, log
import math


def _diff_ProductExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ProductExpression
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
    node: pyomo.core.expr.expr_pyomo5.SumExpression
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
    node: pyomo.core.expr.expr_pyomo5.PowExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 2
    arg1, arg2 = node.args
    der = der_dict[node]
    val1 = val_dict[arg1]
    val2 = val_dict[arg2]
    der_dict[arg1] += der * val2 * val1**(val2 - 1)
    if val1 > 0:
        der_dict[arg2] += der * val1**val2 * log(val1)
    else:
        der_dict[arg2] = math.nan


def _diff_ReciprocalExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.ReciprocalExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] -= der / val_dict[arg]**2


def _diff_exp(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
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
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    der_dict[arg] += der / val_dict[arg]


_unary_map = dict()
_unary_map['exp'] = _diff_exp
_unary_map['log'] = _diff_log


def _diff_UnaryFunctionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    _unary_map[node.getname()](node, val_dict, der_dict)


_diff_map = dict()
_diff_map[_expr.ProductExpression] = _diff_ProductExpression
_diff_map[_expr.ReciprocalExpression] = _diff_ReciprocalExpression
_diff_map[_expr.PowExpression] = _diff_PowExpression
_diff_map[_expr.SumExpression] = _diff_SumExpression
_diff_map[_expr.MonomialTermExpression] = _diff_ProductExpression
_diff_map[_expr.UnaryFunctionExpression] = _diff_UnaryFunctionExpression


class _ReverseADVisitorA(ExpressionValueVisitor):
    def __init__(self, val_dict, der_dict):
        """
        Parameters
        ----------
        val_dict: ComponentMap
        der_dict: ComponentMap
        """
        self.val_dict = val_dict
        self.der_dict = der_dict

    def visit(self, node, values):
        self.val_dict[node] = node._apply_operation(values)
        self.der_dict[node] = 0
        return self.val_dict[node]

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.val_dict[node] = node
            if node not in self.der_dict:
                self.der_dict[node] = 0
            return True, node

        if not node.is_expression_type():
            val = value(node)
            self.val_dict[node] = val
            if node not in self.der_dict:
                self.der_dict[node] = 0
            return True, val

        return False, None


class _ReverseADVisitorB(ExpressionValueVisitor):
    def __init__(self, val_dict, der_dict):
        """
        Parameters
        ----------
        val_dict: ComponentMap
        der_dict: ComponentMap
        """
        self.val_dict = val_dict
        self.der_dict = der_dict

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, None

        if not node.is_expression_type():
            return True, None

        _diff_map[node.__class__](node, self.val_dict, self.der_dict)

        return False, None


def reverse_ad(expr):
    """
    First order reverse ad

    Parameters
    ----------
    expr: pyomo.core.expr.expr_pyomo5.ExpressionBase
        expression to differentiate

    Returns
    -------
    pyomo.core.kernel.component_map.ComponentMap
        component_map mapping variables to derivatives with respect to the corresponding variable
    """
    val_dict = ComponentMap()
    der_dict = ComponentMap()

    visitorA = _ReverseADVisitorA(val_dict, der_dict)
    visitorA.dfs_postorder_stack(expr)
    der_dict[expr] = 1
    visitorB = _ReverseADVisitorB(val_dict, der_dict)
    visitorB.dfs_postorder_stack(expr)

    return der_dict


