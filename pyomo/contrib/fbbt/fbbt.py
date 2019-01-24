from pyomo.core.kernel.component_map import ComponentMap
import pyomo.core.expr.expr_pyomo5 as _expr
from pyomo.core.expr.expr_pyomo5 import ExpressionValueVisitor, nonpyomo_leaf_types, value
from pyomo.core.expr.current import exp, log, sin, cos, tan, asin, acos, atan
import pyomo.contrib.fbbt.interval as interval
import math


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
    bnds_dict[node] = (-1, 1)


def _prop_bnds_leaf_to_root_cos(node, bnds_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    bnds_dict: ComponentMap
    """
    bnds_dict[node] = (-1, 1)


def _prop_bnds_leaf_to_root_tan(node, lb_dict, ub_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    lb_dict: ComponentMap
    ub_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = ub_dict[node]
    ub_dict[arg] += der / (cos(lb_dict[arg])**2)


def _prop_bnds_leaf_to_root_asin(node, lb_dict, ub_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    lb_dict: ComponentMap
    ub_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = ub_dict[node]
    ub_dict[arg] += der / (1 - lb_dict[arg]**2)**0.5


def _prop_bnds_leaf_to_root_acos(node, lb_dict, ub_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    lb_dict: ComponentMap
    ub_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = ub_dict[node]
    ub_dict[arg] -= der / (1 - lb_dict[arg]**2)**0.5


def _prop_bnds_leaf_to_root_atan(node, lb_dict, ub_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    lb_dict: ComponentMap
    ub_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = ub_dict[node]
    ub_dict[arg] += der / (1 + lb_dict[arg]**2)


_unary_map = dict()
_unary_map['exp'] = _prop_bnds_leaf_to_root_exp
_unary_map['log'] = _prop_bnds_leaf_to_root_log
_unary_map['sin'] = _prop_bnds_leaf_to_root_sin
_unary_map['cos'] = _prop_bnds_leaf_to_root_cos
_unary_map['tan'] = _prop_bnds_leaf_to_root_tan
_unary_map['asin'] = _prop_bnds_leaf_to_root_asin
_unary_map['acos'] = _prop_bnds_leaf_to_root_acos
_unary_map['atan'] = _prop_bnds_leaf_to_root_atan


def _prop_bnds_leaf_to_root_UnaryFunctionExpression(node, lb_dict, ub_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    lb_dict: ComponentMap
    ub_dict: ComponentMap
    """
    if node.getname() in _unary_map:
        _unary_map[node.getname()](node, lb_dict, ub_dict)
    else:
        raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))


_prop_bnds_leaf_to_root_map = dict()
_prop_bnds_leaf_to_root_map[_expr.ProductExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[_expr.ReciprocalExpression] = _prop_bnds_leaf_to_root_ReciprocalExpression
_prop_bnds_leaf_to_root_map[_expr.PowExpression] = _prop_bnds_leaf_to_root_PowExpression
_prop_bnds_leaf_to_root_map[_expr.SumExpression] = _prop_bnds_leaf_to_root_SumExpression
_prop_bnds_leaf_to_root_map[_expr.MonomialTermExpression] = _prop_bnds_leaf_to_root_ProductExpression
_prop_bnds_leaf_to_root_map[_expr.NegationExpression] = _prop_bnds_leaf_to_root_NegationExpression
_prop_bnds_leaf_to_root_map[_expr.UnaryFunctionExpression] = _prop_bnds_leaf_to_root_UnaryFunctionExpression


class _ReverseADVisitorA(ExpressionValueVisitor):
    def __init__(self, lb_dict, ub_dict):
        """
        Parameters
        ----------
        lb_dict: ComponentMap
        ub_dict: ComponentMap
        """
        self.lb_dict = lb_dict
        self.ub_dict = ub_dict

    def visit(self, node, values):
        self.lb_dict[node] = node._apply_operation(values)
        self.ub_dict[node] = 0
        return self.lb_dict[node]

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.lb_dict[node] = node
            if node not in self.ub_dict:
                self.ub_dict[node] = 0
            return True, node

        if not node.is_expression_type():
            val = value(node)
            self.lb_dict[node] = val
            if node not in self.ub_dict:
                self.ub_dict[node] = 0
            return True, val

        return False, None


class _ReverseADVisitorB(ExpressionValueVisitor):
    def __init__(self, lb_dict, ub_dict):
        """
        Parameters
        ----------
        lb_dict: ComponentMap
        ub_dict: ComponentMap
        """
        self.lb_dict = lb_dict
        self.ub_dict = ub_dict

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, None

        if not node.is_expression_type():
            return True, None

        if node.__class__ in _prop_bnds_leaf_to_root_map:
            _prop_bnds_leaf_to_root_map[node.__class__](node, self.lb_dict, self.ub_dict)
        else:
            raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))

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
    lb_dict = ComponentMap()
    ub_dict = ComponentMap()

    visitorA = _ReverseADVisitorA(lb_dict, ub_dict)
    visitorA.dfs_postorder_stack(expr)
    ub_dict[expr] = 1
    visitorB = _ReverseADVisitorB(lb_dict, ub_dict)
    visitorB.dfs_postorder_stack(expr)

    return ub_dict


class _ReverseSDVisitorA(ExpressionValueVisitor):
    def __init__(self, lb_dict, ub_dict):
        """
        Parameters
        ----------
        lb_dict: ComponentMap
        ub_dict: ComponentMap
        """
        self.lb_dict = lb_dict
        self.ub_dict = ub_dict

    def visit(self, node, values):
        self.lb_dict[node] = node.create_node_with_local_data(values)
        self.ub_dict[node] = 0
        return self.lb_dict[node]

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.lb_dict[node] = node
            if node not in self.ub_dict:
                self.ub_dict[node] = 0
            return True, node

        if not node.is_expression_type():
            val = node
            self.lb_dict[node] = val
            if node not in self.ub_dict:
                self.ub_dict[node] = 0
            return True, val

        return False, None


class _ReverseSDVisitorB(ExpressionValueVisitor):
    def __init__(self, lb_dict, ub_dict):
        """
        Parameters
        ----------
        lb_dict: ComponentMap
        ub_dict: ComponentMap
        """
        self.lb_dict = lb_dict
        self.ub_dict = ub_dict

    def visit(self, node, values):
        pass

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, None

        if not node.is_expression_type():
            return True, None

        if node.__class__ in _prop_bnds_leaf_to_root_map:
            _prop_bnds_leaf_to_root_map[node.__class__](node, self.lb_dict, self.ub_dict)
        else:
            raise FBBTException('Unsupported expression type for FBBT: {0}'.format(type(node)))

        return False, None


def reverse_sd(expr):
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
    lb_dict = ComponentMap()
    ub_dict = ComponentMap()

    visitorA = _ReverseSDVisitorA(lb_dict, ub_dict)
    visitorA.dfs_postorder_stack(expr)
    ub_dict[expr] = 1
    visitorB = _ReverseSDVisitorB(lb_dict, ub_dict)
    visitorB.dfs_postorder_stack(expr)

    return ub_dict


