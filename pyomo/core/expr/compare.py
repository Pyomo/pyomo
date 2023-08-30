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
import collections
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from pyomo.core.expr import (
    LinearExpression,
    MonomialTermExpression,
    SumExpression,
    ExpressionBase,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    NegationExpression,
    UnaryFunctionExpression,
    ExternalFunctionExpression,
    NPV_ProductExpression,
    NPV_DivisionExpression,
    NPV_PowExpression,
    NPV_SumExpression,
    NPV_NegationExpression,
    NPV_UnaryFunctionExpression,
    NPV_ExternalFunctionExpression,
    Expr_ifExpression,
    AbsExpression,
    NPV_AbsExpression,
    NumericValue,
    RangedExpression,
    InequalityExpression,
    EqualityExpression,
    GetItemExpression,
)
from typing import List
from pyomo.common.collections import Sequence
from pyomo.common.errors import PyomoException
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import native_types


def handle_expression(node: ExpressionBase, pn: List):
    pn.append((type(node), node.nargs()))
    return node.args


def handle_named_expression(node, pn: List, include_named_exprs=True):
    if include_named_exprs:
        pn.append((type(node), 1))
    return (node.expr,)


def handle_unary_expression(node: UnaryFunctionExpression, pn: List):
    pn.append((type(node), 1, node.getname()))
    return node.args


def handle_external_function_expression(node: ExternalFunctionExpression, pn: List):
    pn.append((type(node), node.nargs(), node._fcn))
    return node.args


def _generic_expression_handler():
    return handle_expression


handler = collections.defaultdict(_generic_expression_handler)

handler[UnaryFunctionExpression] = handle_unary_expression
handler[NPV_UnaryFunctionExpression] = handle_unary_expression
handler[ExternalFunctionExpression] = handle_external_function_expression
handler[NPV_ExternalFunctionExpression] = handle_external_function_expression
handler[AbsExpression] = handle_unary_expression
handler[NPV_AbsExpression] = handle_unary_expression
handler[RangedExpression] = handle_expression


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
                return (
                    handle_named_expression(
                        node, self._result, self._include_named_exprs
                    ),
                    None,
                )
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
    if isinstance(expr, Sequence):
        return expr.__class__(visitor.walk_expression(e) for e in expr)
    else:
        return visitor.walk_expression(expr)


def compare_expressions(expr1, expr2, include_named_exprs=True):
    """Returns True if 2 expression trees are identical, False otherwise.

    Parameters
    ----------
    expr1: NumericValue
        A Pyomo Var, Param, or expression
    expr2: NumericValue
        A Pyomo Var, Param, or expression
    include_named_exprs: bool
        If False, then named expressions will be ignored. In other
        words, this function will return True if one expression has a
        named expression and the other does not as long as the rest of
        the expression trees are identical.

    Returns
    -------
    res: bool
        A bool indicating whether or not the expressions are identical.

    """
    pn1 = convert_expression_to_prefix_notation(
        expr1, include_named_exprs=include_named_exprs
    )
    pn2 = convert_expression_to_prefix_notation(
        expr2, include_named_exprs=include_named_exprs
    )
    try:
        res = pn1 == pn2
    except PyomoException:
        res = False
    return res


def assertExpressionsEqual(test, a, b, include_named_exprs=True, places=None):
    """unittest-based assertion for comparing expressions

    This converts the expressions `a` and `b` into prefix notation and
    then compares the resulting lists.

    Parameters
    ----------
    test: unittest.TestCase
        The unittest `TestCase` class that is performing the test.

    a: ExpressionBase or native type

    b: ExpressionBase or native type

    include_named_exprs: bool
       If True (the default), the comparison expands all named
       expressions when generating the prefix notation

    places: Number of decimal places required for equality of floating
            point numbers in the expression. If None (the default), the
            expressions must be exactly equal.
    """
    prefix_a = convert_expression_to_prefix_notation(a, include_named_exprs)
    prefix_b = convert_expression_to_prefix_notation(b, include_named_exprs)
    try:
        test.assertEqual(len(prefix_a), len(prefix_b))
        for _a, _b in zip(prefix_a, prefix_b):
            test.assertIs(_a.__class__, _b.__class__)
            if places is None:
                test.assertEqual(_a, _b)
            else:
                test.assertAlmostEqual(_a, _b, places=places)
    except (PyomoException, AssertionError):
        test.fail(
            f"Expressions not equal:\n\t"
            f"{tostr(prefix_a)}\n\t!=\n\t{tostr(prefix_b)}"
        )


def assertExpressionsStructurallyEqual(
    test, a, b, include_named_exprs=True, places=None
):
    """unittest-based assertion for comparing expressions

    This converts the expressions `a` and `b` into prefix notation and
    then compares the resulting lists.  Operators and (non-native type)
    leaf nodes in the prefix representation are converted to strings
    before comparing (so that things like variables can be compared
    across clones or pickles)

    Parameters
    ----------
    test: unittest.TestCase
        The unittest `TestCase` class that is performing the test.

    a: ExpressionBase or native type

    b: ExpressionBase or native type

    include_named_exprs: bool
       If True (the default), the comparison expands all named
       expressions when generating the prefix notation

    """
    prefix_a = convert_expression_to_prefix_notation(a, include_named_exprs)
    prefix_b = convert_expression_to_prefix_notation(b, include_named_exprs)
    # Convert leaf nodes and operators to their string equivalents
    for prefix in (prefix_a, prefix_b):
        for i, v in enumerate(prefix):
            if type(v) in native_types:
                continue
            if type(v) is tuple:
                # This is an expression node.  Most expression nodes are
                # 2-tuples (node type, nargs), but some are 3-tuples
                # with supplemental data.  The biggest problem is
                # external functions, where the third element is the
                # external function.  We need to convert that to a
                # string to support "structural" comparisons.
                if len(v) == 3:
                    prefix[i] = v[:2] + (str(v[2]),)
                continue
            # This should be a leaf node (Var, mutable Param, etc.).
            # Convert to string to support "structural" comparison
            # (e.g., across clones)
            prefix[i] = str(v)
    try:
        test.assertEqual(len(prefix_a), len(prefix_b))
        for _a, _b in zip(prefix_a, prefix_b):
            if _a.__class__ not in native_types and _b.__class__ not in native_types:
                test.assertIs(_a.__class__, _b.__class__)
            if places is None:
                test.assertEqual(_a, _b)
            else:
                test.assertAlmostEqual(_a, _b, places=places)
    except (PyomoException, AssertionError):
        test.fail(
            f"Expressions not structurally equal:\n\t"
            f"{tostr(prefix_a)}\n\t!=\n\t{tostr(prefix_b)}"
        )
