#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from .numeric_expr import (
    LinearExpression, MonomialTermExpression, SumExpression, ExpressionBase,
    ProductExpression, DivisionExpression, PowExpression,
    NegationExpression, UnaryFunctionExpression, ExternalFunctionExpression,
    NPV_ProductExpression, NPV_DivisionExpression, NPV_PowExpression,
    NPV_SumExpression, NPV_NegationExpression, NPV_UnaryFunctionExpression,
    NPV_ExternalFunctionExpression, Expr_ifExpression, AbsExpression,
    NPV_AbsExpression, NumericValue)
from pyomo.core.expr.logical_expr import (
    RangedExpression, InequalityExpression, EqualityExpression
)
from typing import List
from pyomo.common.errors import PyomoException


def handle_linear_expression(node: LinearExpression, pn: List):
    pn.append((type(node), 2*len(node.linear_vars) + 1))
    pn.append(node.constant)
    pn.extend(node.linear_coefs)
    pn.extend(node.linear_vars)
    return tuple()


def handle_expression(node: ExpressionBase, pn: List):
    pn.append((type(node), node.nargs()))
    return node.args


def handle_named_expression(node, pn: List, include_named_exprs=True):
    if include_named_exprs:
        pn.append((type(node), 1))
    return (node.expr, )


def handle_unary_expression(node: UnaryFunctionExpression, pn: List):
    pn.append((type(node), 1, node.getname()))
    return node.args


def handle_external_function_expression(node: ExternalFunctionExpression, pn: List):
    pn.append((type(node), node.nargs(), node._fcn))
    return node.args


handler = dict()
handler[LinearExpression] = handle_linear_expression
handler[SumExpression] = handle_expression
handler[MonomialTermExpression] = handle_expression
handler[ProductExpression] = handle_expression
handler[DivisionExpression] = handle_expression
handler[PowExpression] = handle_expression
handler[NegationExpression] = handle_expression
handler[NPV_ProductExpression] = handle_expression
handler[NPV_DivisionExpression] = handle_expression
handler[NPV_PowExpression] = handle_expression
handler[NPV_SumExpression] = handle_expression
handler[NPV_NegationExpression] = handle_expression
handler[UnaryFunctionExpression] = handle_unary_expression
handler[NPV_UnaryFunctionExpression] = handle_unary_expression
handler[ExternalFunctionExpression] = handle_external_function_expression
handler[NPV_ExternalFunctionExpression] = handle_external_function_expression
handler[Expr_ifExpression] = handle_expression
handler[AbsExpression] = handle_unary_expression
handler[NPV_AbsExpression] = handle_unary_expression
handler[RangedExpression] = handle_expression
handler[InequalityExpression] = handle_expression
handler[EqualityExpression] = handle_expression


class PrefixVisitor(StreamBasedExpressionVisitor):
    def __init__(self, include_named_exprs=True):
        super().__init__()
        self._result = None
        self._include_named_exprs = include_named_exprs

    def initializeWalker(self, expr):
        self._result = []
        return True, None

    def enterNode(self, node):
        ntype = type(node)
        if ntype in nonpyomo_leaf_types:
            self._result.append(node)
            return tuple(), None

        if node.is_expression_type():
            if node.is_named_expression_type():
                return handle_named_expression(node, self._result, self._include_named_exprs), None
            else:
                return handler[ntype](node, self._result), None
        else:
            self._result.append(node)
            return tuple(), None

    def finalizeResult(self, result):
        ans = self._result
        self._result = None
        return ans


def convert_expression_to_prefix_notation(expr, include_named_exprs=True):
    """
    This function converts pyomo expressions to a list that looks very
    much like prefix notation.  The result can be used in equality
    comparisons to compare expression trees.

    Note that the data structure returned by this function might be
    changed in the future. However, we will maintain that the result
    can be used in equality comparisons.

    Also note that the result should really only be used in equality
    comparisons if the equality comparison is expected to return
    True. If the expressions being compared are expected to be
    different, then the equality comparison will often result in an
    error rather than returning False.

    m = ConcreteModel()
    m.x = Var()
    m.y = Var()

    e1 = m.x * m.y
    e2 = m.x * m.y
    e3 = m.x + m.y

    convert_expression_to_prefix_notation(e1) == convert_expression_to_prefix_notation(e2)  # True
    convert_expression_to_prefix_notation(e1) == convert_expression_to_prefix_notation(e3)  # Error

    However, the compare_expressions function can be used:

    compare_expressions(e1, e2)  # True
    compare_expressions(e1, e3)  # False

    Parameters
    ----------
    expr: NumericValue
        A Pyomo expression, Var, or Param

    Returns
    -------
    prefix_notation: list
        The expression in prefix notation

    """
    visitor = PrefixVisitor(include_named_exprs=include_named_exprs)
    return visitor.walk_expression(expr)


def compare_expressions(expr1, expr2, include_named_exprs=True):
    """
    Returns True if 2 expression trees are identical. Returns False
    otherwise.

    Parameters
    ----------
    expr1: NumericValue
        A Pyomo Var, Param, or expression
    expr2: NumericValue
        A Pyomo Var, Param, or expression
    include_named_exprs: bool
        If False, then named expressions will be ignored. In other words, this function
        will return True if one expression has a named expression and the other does not
        as long as the rest of the expression trees are identical.

    Returns
    -------
    res: bool
        A bool indicating whether or not the expressions are identical.

    """
    pn1 = convert_expression_to_prefix_notation(expr1, include_named_exprs=include_named_exprs)
    pn2 = convert_expression_to_prefix_notation(expr2, include_named_exprs=include_named_exprs)
    try:
        res = pn1 == pn2
    except PyomoException:
        res = False
    return res
