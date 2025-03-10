#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from math import pi
from pyomo.common.collections import ComponentMap
from pyomo.contrib.fbbt.interval import (
    BoolFlag,
    eq,
    ineq,
    ranged,
    if_,
    add,
    acos,
    asin,
    atan,
    cos,
    div,
    exp,
    interval_abs,
    log,
    log10,
    mul,
    power,
    sin,
    sub,
    tan,
)
from pyomo.core.base.expression import Expression
from pyomo.core.expr.numeric_expr import (
    NumericExpression,
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    MonomialTermExpression,
    LinearExpression,
    SumExpression,
    ExternalFunctionExpression,
    Expr_ifExpression,
)
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.expr.numvalue import native_numeric_types, native_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.repn.util import BeforeChildDispatcher, ExitNodeDispatcher

inf = float('inf')
logger = logging.getLogger(__name__)


class ExpressionBoundsBeforeChildDispatcher(BeforeChildDispatcher):
    __slots__ = ()

    def __init__(self):
        self[ExternalFunctionExpression] = self._before_external_function

    @staticmethod
    def _before_external_function(visitor, child):
        # [ESJ 10/6/23]: If external functions ever implement callbacks to help with
        # this then this should use them
        return False, (-inf, inf)

    @staticmethod
    def _before_native_numeric(visitor, child):
        return False, (child, child)

    @staticmethod
    def _before_native_logical(visitor, child):
        return False, (BoolFlag(child), BoolFlag(child))

    @staticmethod
    def _before_var(visitor, child):
        leaf_bounds = visitor.leaf_bounds
        if child in leaf_bounds:
            pass
        elif child.is_fixed() and visitor.use_fixed_var_values_as_bounds:
            val = child.value
            try:
                ans = visitor._before_child_handlers[val.__class__](visitor, val)
            except ValueError:
                raise ValueError(
                    "Var '%s' is fixed to None. This value cannot be used to "
                    "calculate bounds." % child.name
                ) from None
            leaf_bounds[child] = ans[1]
            return ans
        else:
            lb = child.lb
            ub = child.ub
            if lb is None:
                lb = -inf
            if ub is None:
                ub = inf
            leaf_bounds[child] = (lb, ub)
        return False, leaf_bounds[child]

    @staticmethod
    def _before_named_expression(visitor, child):
        leaf_bounds = visitor.leaf_bounds
        if child in leaf_bounds:
            return False, leaf_bounds[child]
        else:
            return True, None

    @staticmethod
    def _before_param(visitor, child):
        val = child.value
        return visitor._before_child_handlers[val.__class__](visitor, val)

    @staticmethod
    def _before_string(visitor, child):
        raise ValueError(
            f"{child!r} ({type(child).__name__}) is not a valid numeric type. "
            f"Cannot compute bounds on expression."
        )

    @staticmethod
    def _before_invalid(visitor, child):
        raise ValueError(
            f"{child!r} ({type(child).__name__}) is not a valid numeric type. "
            f"Cannot compute bounds on expression."
        )

    @staticmethod
    def _before_complex(visitor, child):
        raise ValueError(
            f"Cannot compute bounds on expressions containing "
            f"complex numbers. Encountered when processing {child}"
        )

    @staticmethod
    def _before_npv(visitor, child):
        val = value(child)
        return visitor._before_child_handlers[val.__class__](visitor, val)


def _handle_ProductExpression(visitor, node, arg1, arg2):
    if arg1 is arg2:
        return power(*arg1, 2, 2, feasibility_tol=visitor.feasibility_tol)
    return mul(*arg1, *arg2)


def _handle_SumExpression(visitor, node, *args):
    bnds = (0, 0)
    for arg in args:
        bnds = add(*bnds, *arg)
    return bnds


def _handle_DivisionExpression(visitor, node, arg1, arg2):
    return div(*arg1, *arg2, feasibility_tol=visitor.feasibility_tol)


def _handle_PowExpression(visitor, node, arg1, arg2):
    return power(*arg1, *arg2, feasibility_tol=visitor.feasibility_tol)


def _handle_NegationExpression(visitor, node, arg):
    return sub(0, 0, *arg)


def _handle_exp(visitor, node, arg):
    return exp(*arg)


def _handle_log(visitor, node, arg):
    return log(*arg)


def _handle_log10(visitor, node, arg):
    return log10(*arg)


def _handle_sin(visitor, node, arg):
    return sin(*arg)


def _handle_cos(visitor, node, arg):
    return cos(*arg)


def _handle_tan(visitor, node, arg):
    return tan(*arg)


def _handle_asin(visitor, node, arg):
    return asin(*arg, -pi / 2, pi / 2, visitor.feasibility_tol)


def _handle_acos(visitor, node, arg):
    return acos(*arg, 0, pi, visitor.feasibility_tol)


def _handle_atan(visitor, node, arg):
    return atan(*arg, -pi / 2, pi / 2)


def _handle_sqrt(visitor, node, arg):
    return power(*arg, 0.5, 0.5, feasibility_tol=visitor.feasibility_tol)


def _handle_AbsExpression(visitor, node, arg):
    return interval_abs(*arg)


def _handle_UnaryFunctionExpression(visitor, node, arg):
    return _unary_function_dispatcher[node.getname()](visitor, node, arg)


def _handle_named_expression(visitor, node, arg):
    visitor.leaf_bounds[node] = arg
    return arg


def _handle_unknowable_bounds(visitor, node, arg):
    return -inf, inf


def _handle_equality(visitor, node, arg1, arg2):
    return eq(*arg1, *arg2, feasibility_tol=visitor.feasibility_tol)


def _handle_inequality(visitor, node, arg1, arg2):
    return ineq(*arg1, *arg2, feasibility_tol=visitor.feasibility_tol)


def _handle_ranged(visitor, node, arg1, arg2, arg3):
    return ranged(*arg1, *arg2, *arg3, feasibility_tol=visitor.feasibility_tol)


def _handle_expr_if(visitor, node, arg1, arg2, arg3):
    return if_(*arg1, *arg2, *arg3)


_unary_function_dispatcher = {
    'exp': _handle_exp,
    'log': _handle_log,
    'log10': _handle_log10,
    'sin': _handle_sin,
    'cos': _handle_cos,
    'tan': _handle_tan,
    'asin': _handle_asin,
    'acos': _handle_acos,
    'atan': _handle_atan,
    'sqrt': _handle_sqrt,
}


class ExpressionBoundsExitNodeDispatcher(ExitNodeDispatcher):
    def unexpected_expression_type(self, visitor, node, *args):
        if isinstance(node, NumericExpression):
            ans = -inf, inf
        elif isinstance(node, BooleanExpression):
            ans = BoolFlag(False), BoolFlag(True)
        else:
            super().unexpected_expression_type(visitor, node, *args)
        logger.warning(
            f"Unexpected expression node type '{type(node).__name__}' "
            f"found while walking expression tree; returning {ans} "
            "for the expression bounds."
        )
        return ans


class ExpressionBoundsVisitor(StreamBasedExpressionVisitor):
    """
    Walker to calculate bounds on an expression, from leaf to root, with
    caching of terminal node bounds (Vars and Expressions)

    NOTE: If anything changes on the model (e.g., Var bounds, fixing, mutable
    Param values, etc), then you need to either create a new instance of this
    walker, or clear self.leaf_bounds!

    Parameters
    ----------
    leaf_bounds: ComponentMap in which to cache bounds at leaves of the expression
        tree
    feasibility_tol: float, feasibility tolerance for interval arithmetic
        calculations
    use_fixed_var_values_as_bounds: bool, whether or not to use the values of
        fixed Vars as the upper and lower bounds for those Vars or to instead
        ignore fixed status and use the bounds. Set to 'True' if you do not
        anticipate the fixed status of Variables to change for the duration that
        the computed bounds should be valid.
    """

    _before_child_handlers = ExpressionBoundsBeforeChildDispatcher()
    _operator_dispatcher = ExpressionBoundsExitNodeDispatcher(
        {
            ProductExpression: _handle_ProductExpression,
            DivisionExpression: _handle_DivisionExpression,
            PowExpression: _handle_PowExpression,
            AbsExpression: _handle_AbsExpression,
            SumExpression: _handle_SumExpression,
            MonomialTermExpression: _handle_ProductExpression,
            NegationExpression: _handle_NegationExpression,
            UnaryFunctionExpression: _handle_UnaryFunctionExpression,
            LinearExpression: _handle_SumExpression,
            Expression: _handle_named_expression,
            ExternalFunctionExpression: _handle_unknowable_bounds,
            EqualityExpression: _handle_equality,
            InequalityExpression: _handle_inequality,
            RangedExpression: _handle_ranged,
            Expr_ifExpression: _handle_expr_if,
        }
    )

    def __init__(
        self,
        leaf_bounds=None,
        feasibility_tol=1e-8,
        use_fixed_var_values_as_bounds=False,
    ):
        super().__init__()
        self.leaf_bounds = leaf_bounds if leaf_bounds is not None else ComponentMap()
        self.feasibility_tol = feasibility_tol
        self.use_fixed_var_values_as_bounds = use_fixed_var_values_as_bounds

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, result
        return True, expr

    def beforeChild(self, node, child, child_idx):
        return self._before_child_handlers[child.__class__](self, child)

    def exitNode(self, node, data):
        return self._operator_dispatcher[node.__class__](self, node, *data)
