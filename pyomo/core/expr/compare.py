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
from .numeric_expr import (LinearExpression, MonomialTermExpression, SumExpression, ExpressionBase,
                           ProductExpression, DivisionExpression, ReciprocalExpression, PowExpression,
                           NegationExpression, UnaryFunctionExpression, ExternalFunctionExpression,
                           NPV_ProductExpression, NPV_DivisionExpression, NPV_ReciprocalExpression,
                           NPV_PowExpression, NPV_SumExpression, NPV_NegationExpression,
                           NPV_UnaryFunctionExpression, NPV_ExternalFunctionExpression, Expr_ifExpression,
                           AbsExpression, NPV_AbsExpression)
from pyomo.core.expr.logical_expr import RangedExpression, InequalityExpression, EqualityExpression
from typing import List
from pyomo.common.errors import PyomoException


def handle_linear_expression(node: LinearExpression, pn: List):
    pn.append((LinearExpression, 2*len(node.linear_vars) + 1))
    pn.append(node.constant)
    pn.extend(node.linear_coefs)
    pn.extend(node.linear_vars)
    return tuple()


def handle_expression(node: ExpressionBase, pn: List):
    pn.append((type(node), node.nargs()))
    return node.args


def handle_unary_expression(node: UnaryFunctionExpression, pn: List):
    pn.append((UnaryFunctionExpression, 1, node.getname()))
    return node.args


def handle_external_function_expression(node: ExternalFunctionExpression, pn: List):
    pn.append((ExternalFunctionExpression, node.nargs(), node._fcn))
    return node.args


handler = dict()
handler[LinearExpression] = handle_linear_expression
handler[SumExpression] = handle_expression
handler[MonomialTermExpression] = handle_expression
handler[ProductExpression] = handle_expression
handler[DivisionExpression] = handle_expression
handler[ReciprocalExpression] = handle_expression
handler[PowExpression] = handle_expression
handler[NegationExpression] = handle_expression
handler[NPV_ProductExpression] = handle_expression
handler[NPV_DivisionExpression] = handle_expression
handler[NPV_ReciprocalExpression] = handle_expression
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
    def __init__(self):
        super().__init__()
        self._result = list()

    def initializeWalker(self, expr):
        self._result = list()
        etype = type(expr)

        if etype in nonpyomo_leaf_types:
            self._result.append(expr)
            return False, self._result

        if expr.is_expression_type():
            if etype is LinearExpression:
                handle_linear_expression(expr, self._result)
                return False, self._result
            return True, None

        self._result.append(expr)
        return False, self._result

    def enterNode(self, node):
        ntype = type(node)
        if ntype in nonpyomo_leaf_types:
            self._result.append(node)
            return tuple(), None

        if node.is_expression_type():
            return handler[ntype](node, self._result), None
        else:
            self._result.append(node)
            return tuple(), None

    def finalizeResult(self, result):
        return self._result


def convert_expression_to_prefix_notation(expr):
    visitor = PrefixVisitor()
    return visitor.walk_expression(expr)


def compare_expressions(expr1, expr2):
    pn1 = convert_expression_to_prefix_notation(expr1)
    pn2 = convert_expression_to_prefix_notation(expr2)
    try:
        res = pn1 == pn2
    except PyomoException:
        res = False
    return res
