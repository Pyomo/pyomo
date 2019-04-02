from pyomo.core.base.expression import SimpleExpression, _GeneralExpressionData
from pyomo.core.kernel.component_map import ComponentMap
import pyomo.core.expr.expr_pyomo5 as _expr
from pyomo.core.expr.expr_pyomo5 import ExpressionValueVisitor, nonpyomo_leaf_types, value
from pyomo.core.expr.current import exp, log, sin, cos, tan, asin, acos, atan


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
    if arg2.__class__ not in nonpyomo_leaf_types:
        der_dict[arg2] += der * val1**val2 * log(val1)


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


def _diff_NegationExpression(node, val_dict, der_dict):
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
    der_dict[arg] -= der


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


def _diff_sin(node, val_dict, der_dict):
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
    der_dict[arg] += der * cos(val_dict[arg])


def _diff_cos(node, val_dict, der_dict):
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
    der_dict[arg] -= der * sin(val_dict[arg])


def _diff_tan(node, val_dict, der_dict):
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
    der_dict[arg] += der / (cos(val_dict[arg])**2)


def _diff_asin(node, val_dict, der_dict):
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
    der_dict[arg] += der / (1 - val_dict[arg]**2)**0.5


def _diff_acos(node, val_dict, der_dict):
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
    der_dict[arg] -= der / (1 - val_dict[arg]**2)**0.5


def _diff_atan(node, val_dict, der_dict):
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
    der_dict[arg] += der / (1 + val_dict[arg]**2)


_unary_map = dict()
_unary_map['exp'] = _diff_exp
_unary_map['log'] = _diff_log
_unary_map['sin'] = _diff_sin
_unary_map['cos'] = _diff_cos
_unary_map['tan'] = _diff_tan
_unary_map['asin'] = _diff_asin
_unary_map['acos'] = _diff_acos
_unary_map['atan'] = _diff_atan


def _diff_UnaryFunctionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.expr_pyomo5.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    if node.getname() in _unary_map:
        _unary_map[node.getname()](node, val_dict, der_dict)
    else:
        raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))


def _diff_SimpleExpression(node, val_dict, der_dict):
    der = der_dict[node]
    der_dict[node.expr] += der


_diff_map = dict()
_diff_map[_expr.ProductExpression] = _diff_ProductExpression
_diff_map[_expr.ReciprocalExpression] = _diff_ReciprocalExpression
_diff_map[_expr.PowExpression] = _diff_PowExpression
_diff_map[_expr.SumExpression] = _diff_SumExpression
_diff_map[_expr.MonomialTermExpression] = _diff_ProductExpression
_diff_map[_expr.NegationExpression] = _diff_NegationExpression
_diff_map[_expr.UnaryFunctionExpression] = _diff_UnaryFunctionExpression
_diff_map[SimpleExpression] = _diff_SimpleExpression
_diff_map[_GeneralExpressionData] = _diff_SimpleExpression


class _ReverseADVisitorLeafToRoot(ExpressionValueVisitor):
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


class _ReverseADVisitorRootToLeaf(ExpressionValueVisitor):
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

        if node.__class__ in _diff_map:
            _diff_map[node.__class__](node, self.val_dict, self.der_dict)
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
        component_map mapping variables to derivatives with respect
        to the corresponding variable
    """
    val_dict = ComponentMap()
    der_dict = ComponentMap()

    visitorA = _ReverseADVisitorLeafToRoot(val_dict, der_dict)
    visitorA.dfs_postorder_stack(expr)
    der_dict[expr] = 1
    visitorB = _ReverseADVisitorRootToLeaf(val_dict, der_dict)
    visitorB.dfs_postorder_stack(expr)

    return der_dict


class _ReverseSDVisitorLeafToRoot(ExpressionValueVisitor):
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
        self.val_dict[node] = node.create_node_with_local_data(values)
        self.der_dict[node] = 0
        return self.val_dict[node]

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.val_dict[node] = node
            if node not in self.der_dict:
                self.der_dict[node] = 0
            return True, node

        if not node.is_expression_type():
            val = node
            self.val_dict[node] = val
            if node not in self.der_dict:
                self.der_dict[node] = 0
            return True, val

        return False, None


class _ReverseSDVisitorRootToLeaf(ExpressionValueVisitor):
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

        if node.__class__ in _diff_map:
            _diff_map[node.__class__](node, self.val_dict, self.der_dict)
        else:
            raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))

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
        component_map mapping variables to derivatives with respect
        to the corresponding variable
    """
    val_dict = ComponentMap()
    der_dict = ComponentMap()

    visitorA = _ReverseSDVisitorLeafToRoot(val_dict, der_dict)
    visitorA.dfs_postorder_stack(expr)
    der_dict[expr] = 1
    visitorB = _ReverseSDVisitorRootToLeaf(val_dict, der_dict)
    visitorB.dfs_postorder_stack(expr)

    return der_dict


