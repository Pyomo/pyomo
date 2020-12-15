#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentMap
from pyomo.core.expr import current as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.current import exp, log, sin, cos
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


def _diff_LinearExpression(node, val_dict, der_dict):
    der = der_dict[node]
    for ndx, v in enumerate(node.linear_vars):
        coef = node.linear_coefs[ndx]
        der_dict[v] += der * val_dict[coef]
        der_dict[coef] += der * val_dict[v]

    der_dict[node.constant] += der

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
    der_dict[arg1] += der * val2 * val1**(val2 - 1)
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
    der_dict[num] += der * (1/val_dict[den])
    der_dict[den] -= der * val_dict[num] / val_dict[den]**2


def _diff_ReciprocalExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ReciprocalExpression
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
    der_dict[arg] += der / (cos(val_dict[arg])**2)


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
    der_dict[arg] += der / (1 - val_dict[arg]**2)**0.5


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
    der_dict[arg] -= der / (1 - val_dict[arg]**2)**0.5


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
    der_dict[arg] += der / (1 + val_dict[arg]**2)


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
    der_dict[arg] += der * 0.5 * val_dict[arg]**(-0.5)


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
        raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))


def _diff_ExternalFunctionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    der = der_dict[node]
    vals = tuple(val_dict[i] for i in node.args)
    derivs = node._fcn.evaluate_fgh(vals)[1]
    for ndx, arg in enumerate(node.args):
        der_dict[arg] += der * derivs[ndx]


_diff_map = dict()
_diff_map[_expr.ProductExpression] = _diff_ProductExpression
_diff_map[_expr.DivisionExpression] = _diff_DivisionExpression
_diff_map[_expr.ReciprocalExpression] = _diff_ReciprocalExpression
_diff_map[_expr.PowExpression] = _diff_PowExpression
_diff_map[_expr.SumExpression] = _diff_SumExpression
_diff_map[_expr.MonomialTermExpression] = _diff_ProductExpression
_diff_map[_expr.NegationExpression] = _diff_NegationExpression
_diff_map[_expr.UnaryFunctionExpression] = _diff_UnaryFunctionExpression
_diff_map[_expr.ExternalFunctionExpression] = _diff_ExternalFunctionExpression
_diff_map[_expr.LinearExpression] = _diff_LinearExpression

_diff_map[_expr.NPV_ProductExpression] = _diff_ProductExpression
_diff_map[_expr.NPV_DivisionExpression] = _diff_DivisionExpression
_diff_map[_expr.NPV_ReciprocalExpression] = _diff_ReciprocalExpression
_diff_map[_expr.NPV_PowExpression] = _diff_PowExpression
_diff_map[_expr.NPV_SumExpression] = _diff_SumExpression
_diff_map[_expr.NPV_NegationExpression] = _diff_NegationExpression
_diff_map[_expr.NPV_UnaryFunctionExpression] = _diff_UnaryFunctionExpression
_diff_map[_expr.NPV_ExternalFunctionExpression] = _diff_ExternalFunctionExpression


class _NamedExpressionCollector(ExpressionValueVisitor):
    def __init__(self):
        self.named_expressions = list()

    def visit(self, node, values):
        return None

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, None

        if not node.is_expression_type():
            return True, None

        if node.is_named_expression_type():
            self.named_expressions.append(node)
            return False, None

        return False, None


def _collect_ordered_named_expressions(expr):
    """
    The purpose of this function is to collect named expressions in a
    particular order. The order is very important. In the resulting
    list each named expression can only appear once, and any named
    expressions that are used in other named expressions have to come
    after the named expression that use them.
    """    
    visitor = _NamedExpressionCollector()
    visitor.dfs_postorder_stack(expr)
    named_expressions = visitor.named_expressions
    seen = set()
    res = list()
    for e in reversed(named_expressions):
        if id(e) in seen:
            continue
        seen.add(id(e))
        res.append(e)
    res = list(reversed(res))
    return res


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

        if node.__class__ is _expr.LinearExpression:
            for v in node.linear_vars + node.linear_coefs + [node.constant]:
                val = value(v)
                self.val_dict[v] = val
                if v not in self.der_dict:
                    self.der_dict[v] = 0
            val = value(node)
            self.val_dict[node] = val
            if node not in self.der_dict:
                self.der_dict[node] = 0
            return True, val

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

        if node.is_named_expression_type():
            return True, None

        if node.__class__ in _diff_map:
            _diff_map[node.__class__](node, self.val_dict, self.der_dict)
            return False, None
        else:
            raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))


def reverse_ad(expr):
    """
    First order reverse ad

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.ExpressionBase
        expression to differentiate

    Returns
    -------
    ComponentMap
        component_map mapping variables to derivatives with respect
        to the corresponding variable
    """
    val_dict = ComponentMap()
    der_dict = ComponentMap()

    visitorA = _ReverseADVisitorLeafToRoot(val_dict, der_dict)
    visitorA.dfs_postorder_stack(expr)
    named_expressions = _collect_ordered_named_expressions(expr)
    der_dict[expr] = 1
    visitorB = _ReverseADVisitorRootToLeaf(val_dict, der_dict)
    visitorB.dfs_postorder_stack(expr)
    for named_expr in named_expressions:
        der_dict[named_expr.expr] = der_dict[named_expr]
        visitorB.dfs_postorder_stack(named_expr.expr)

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
        self.val_dict[node] = node.create_node_with_local_data(tuple(values))
        self.der_dict[node] = 0
        return self.val_dict[node]

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            self.val_dict[node] = node
            if node not in self.der_dict:
                self.der_dict[node] = 0
            return True, node

        if node.__class__ is _expr.LinearExpression:
            for v in node.linear_vars + node.linear_coefs + [node.constant]:
                val = v
                self.val_dict[v] = val
                if v not in self.der_dict:
                    self.der_dict[v] = 0
            val = node
            self.val_dict[node] = val
            if node not in self.der_dict:
                self.der_dict[node] = 0
            return True, val

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

        if node.is_named_expression_type():
            return True, None

        if node.__class__ in _diff_map:
            _diff_map[node.__class__](node, self.val_dict, self.der_dict)
            return False, None
        else:
            raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))


def reverse_sd(expr):
    """
    First order reverse ad

    Parameters
    ----------
    expr: pyomo.core.expr.numeric_expr.ExpressionBase
        expression to differentiate

    Returns
    -------
    ComponentMap
        component_map mapping variables to derivatives with respect
        to the corresponding variable
    """
    val_dict = ComponentMap()
    der_dict = ComponentMap()

    visitorA = _ReverseSDVisitorLeafToRoot(val_dict, der_dict)
    visitorA.dfs_postorder_stack(expr)
    named_expressions = _collect_ordered_named_expressions(expr)
    der_dict[expr] = 1
    visitorB = _ReverseSDVisitorRootToLeaf(val_dict, der_dict)
    visitorB.dfs_postorder_stack(expr)
    for named_expr in named_expressions:
        der_dict[named_expr.expr] = der_dict[named_expr]
        visitorB.dfs_postorder_stack(named_expr.expr)

    return der_dict
