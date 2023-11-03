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

from math import pi
from pyomo.common.collections import ComponentMap
from pyomo.contrib.fbbt.interval import (
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
)
from pyomo.core.expr.numvalue import native_numeric_types, native_types, value
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.repn.util import BeforeChildDispatcher, ExitNodeDispatcher

inf = float('inf')


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
    def _before_var(visitor, child):
        leaf_bounds = visitor.leaf_bounds
        if child in leaf_bounds:
            pass
        elif child.is_fixed() and visitor.use_fixed_var_values_as_bounds:
            val = child.value
            if val is None:
                raise ValueError(
                    "Var '%s' is fixed to None. This value cannot be used to "
                    "calculate bounds." % child.name
                )
            leaf_bounds[child] = (child.value, child.value)
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
        return False, (child.value, child.value)

    @staticmethod
    def _before_native(visitor, child):
        return False, (child, child)

    @staticmethod
    def _before_string(visitor, child):
        raise ValueError(
            f"{child!r} ({type(child)}) is not a valid numeric type. "
            f"Cannot compute bounds on expression."
        )

    @staticmethod
    def _before_invalid(visitor, child):
        raise ValueError(
            f"{child!r} ({type(child)}) is not a valid numeric type. "
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
        return False, (val, val)


_before_child_handlers = ExpressionBoundsBeforeChildDispatcher()


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


_operator_dispatcher = ExitNodeDispatcher(
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
    }
)


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
        return _before_child_handlers[child.__class__](self, child)

    def exitNode(self, node, data):
        return _operator_dispatcher[node.__class__](self, node, *data)
