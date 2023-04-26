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

import copy

from pyomo.core.expr.current import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    Expr_ifExpression,
)
from pyomo.core.base.expression import ScalarExpression
from . import linear

_CONSTANT = linear._CONSTANT
_LINEAR = linear._LINEAR
_GENERAL = linear._GENERAL


class _QUADRATIC(object):
    pass


class QuadraticRepn(object):
    __slots__ = ("multiplier", "constant", "linear", "quadratic", "nonlinear")

    def __init__(self):
        self.multiplier = 1
        self.constant = 0
        self.linear = {}
        self.quadratic = None
        self.nonlinear = None

    def __str__(self):
        return (
            f"QuadraticRepn(mult={self.multiplier}, const={self.constant}, "
            f"linear={self.linear}, quadratic={self.quadratic}, "
            "nonlinear={self.nonlinear})"
        )

    def __repr__(self):
        return str(self)

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return _GENERAL, self
        elif self.quadratic:
            return _QUADRATIC, self
        elif self.linear:
            return _LINEAR, self
        else:
            return _CONSTANT, self.multiplier * self.constant

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.multiplier = self.multiplier
        ans.constant = self.constant
        ans.linear = dict(self.linear)
        if self.quadratic:
            ans.quadratic = dict(self.quadratic)
        ans.nonlinear = self.nonlinear
        return ans

    def to_expression(visitor):
        var_map = visitor.var_map
        if self.linear:
            ans = (
                LinearExpression(
                    [
                        MonomialTermExpression((coef, var_map[vid]))
                        for vid, coef in self.linear.items()
                        if coef
                    ]
                )
                + self.constant
            )
        else:
            ans = self.constant
        if self.quadratic:
            with mutable_expression() as e:
                for (x1, x2), coef in self.quadratic.items():
                    e += var_map[x1] * var_map[x2] * coef
                e += ans
            ans = e
        if self.nonlinear is not None:
            ans += self.nonlinear
        if self.multiplier != 1:
            ans *= self.multiplier
        return ans

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a QuadraticRepn() as a data object in
        the expression walker (thereby avoiding the function call for a
        custom callback)

        """
        # Note that self.multiplier will always be 1 (we only call append()
        # within a sum, so there is no opportunity for self.multiplier to
        # change). Omitting the assertion for efficiency.
        # assert self.multiplier == 1
        _type, other = other
        if _type is _CONSTANT:
            self.constant += other
            return

        mult = other.multiplier
        self.constant += mult * other.constant
        if other.linear:
            linear._merge_dict(mult, self.linear, other.linear)
        if other.quadratic:
            if not self.quadratic:
                self.quadratic = {}
            linear._merge_dict(mult, self.quadratic, other.quadratic)
        if other.nonlinear is not None:
            if mult != 1:
                nl = mult * other.nonlinear
            else:
                nl = other.nonlinear
            if self.nonlinear is None:
                self.nonlinear = nl
            else:
                self.nonlinear += nl


_exit_node_handlers = copy.deepcopy(linear._exit_node_handlers)

#
# NEGATION
#
_exit_node_handlers[NegationExpression][(_QUADRATIC,)] = linear._handle_negation_ANY


#
# PRODUCT
#
def _handle_product_linear_linear(visitor, node, arg1, arg2):
    _, arg1 = arg1
    _, arg2 = arg2
    # Quadratic first, because we will update linear in a minute
    quadratic = arg1.quadratic = {}
    for vid1, coef1 in arg1.linear.items():
        for vid2, coef2 in arg2.linear.items():
            if vid1 < vid2:
                key = vid1, vid2
            else:
                key = vid2, vid1
            if key in quadratic:
                quadratic[key] += coef1 * coef2
            else:
                quadratic[key] = coef1 * coef2
    # Linear second, as this relies on knowing the original constants
    if not arg2.constant:
        arg1.linear = {}
    elif arg2.constant != 1:
        c = arg2.constant
        linear = arg1.linear
        for vid, coef in linear.items():
            linear[k] = c * coef
    if arg1.constant:
        linear._merge_dict(arg1.constant, arg1.linear, arg2.linear)
    # Finally, the constant and multipliers
    arg1.constant *= arg2.constant
    arg1.multiplier *= arg2.multiplier
    return _QUADRATIC, arg1


_exit_node_handlers[ProductExpression].update(
    {
        (_CONSTANT, _QUADRATIC): linear._handle_product_constant_ANY,
        (_LINEAR, _QUADRATIC): linear._handle_product_nonlinear,
        (_QUADRATIC, _QUADRATIC): linear._handle_product_nonlinear,
        (_GENERAL, _QUADRATIC): linear._handle_product_nonlinear,
        (_QUADRATIC, _CONSTANT): linear._handle_product_ANY_constant,
        (_QUADRATIC, _LINEAR): linear._handle_product_nonlinear,
        (_QUADRATIC, _GENERAL): linear._handle_product_nonlinear,
        (_LINEAR, _LINEAR): _handle_product_linear_linear,
    }
)

#
# DIVISION
#
_exit_node_handlers[DivisionExpression].update(
    {
        (_CONSTANT, _QUADRATIC): linear._handle_division_nonlinear,
        (_LINEAR, _QUADRATIC): linear._handle_division_nonlinear,
        (_QUADRATIC, _QUADRATIC): linear._handle_division_nonlinear,
        (_GENERAL, _QUADRATIC): linear._handle_division_nonlinear,
        (_QUADRATIC, _CONSTANT): linear._handle_division_ANY_constant,
        (_QUADRATIC, _LINEAR): linear._handle_division_nonlinear,
        (_QUADRATIC, _GENERAL): linear._handle_division_nonlinear,
    }
)

#
# EXPONENTIATION
#
_exit_node_handlers[PowExpression].update(
    {
        (_CONSTANT, _QUADRATIC): linear._handle_pow_nonlinear,
        (_LINEAR, _QUADRATIC): linear._handle_pow_nonlinear,
        (_QUADRATIC, _QUADRATIC): linear._handle_pow_nonlinear,
        (_GENERAL, _QUADRATIC): linear._handle_pow_nonlinear,
        (_QUADRATIC, _CONSTANT): linear._handle_pow_ANY_constant,
        (_QUADRATIC, _LINEAR): linear._handle_pow_nonlinear,
        (_QUADRATIC, _GENERAL): linear._handle_pow_nonlinear,
    }
)

#
# ABS and UNARY handlers
#
_exit_node_handlers[AbsExpression][(_QUADRATIC,)] = linear._handle_unary_nonlinear
_exit_node_handlers[UnaryFunctionExpression][
    (_QUADRATIC,)
] = linear._handle_unary_nonlinear

#
# NAMED EXPRESSION handlers
#
_exit_node_handlers[ScalarExpression][(_QUADRATIC,)] = linear._handle_named_ANY

#
# EXPR_IF handlers
#
# Note: it is easier to just recreate the entire data structure, rather
# than update it
_exit_node_handlers[Expr_ifExpression] = {
    (i, j, k): linear._handle_expr_if_nonlinear
    for i in (_LINEAR, _QUADRATIC, _GENERAL)
    for j in (_CONSTANT, _LINEAR, _QUADRATIC, _GENERAL)
    for k in (_CONSTANT, _LINEAR, _QUADRATIC, _GENERAL)
}
for j in (_CONSTANT, _LINEAR, _QUADRATIC, _GENERAL):
    for k in (_CONSTANT, _LINEAR, _QUADRATIC, _GENERAL):
        _exit_node_handlers[Expr_ifExpression][
            _CONSTANT, j, k
        ] = linear._handle_expr_if_const


class QuadraticRepnVisitor(linear.LinearRepnVisitor):
    Result = QuadraticRepn
    exit_node_handlers = _exit_node_handlers
    exit_node_dispatcher = linear._initialize_exit_node_dispatcher(_exit_node_handlers)
