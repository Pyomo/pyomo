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

from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    Expr_ifExpression,
    LinearExpression,
    MonomialTermExpression,
    mutable_expression,
)
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.base.expression import Expression
from . import linear
from .linear import _merge_dict, to_expression

_CONSTANT = linear.ExprType.CONSTANT
_LINEAR = linear.ExprType.LINEAR
_GENERAL = linear.ExprType.GENERAL
_QUADRATIC = linear.ExprType.QUADRATIC


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
            f"nonlinear={self.nonlinear})"
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
        else:
            ans.quadratic = None
        ans.nonlinear = self.nonlinear
        return ans

    def to_expression(self, visitor):
        var_map = visitor.var_map
        if self.nonlinear is not None:
            # We want to start with the nonlinear term (and use
            # assignment) in case the term is a non-numeric node (like a
            # relational expression)
            ans = self.nonlinear
        else:
            ans = 0
        if self.quadratic:
            with mutable_expression() as e:
                for (x1, x2), coef in self.quadratic.items():
                    if x1 == x2:
                        e += coef * var_map[x1] ** 2
                    else:
                        e += coef * (var_map[x1] * var_map[x2])
            ans += e
        if self.linear:
            if len(self.linear) == 1:
                vid, coef = next(iter(self.linear.items()))
                if coef == 1:
                    ans += var_map[vid]
                elif coef:
                    ans += MonomialTermExpression((coef, var_map[vid]))
                else:
                    pass
            else:
                ans += LinearExpression(
                    [
                        MonomialTermExpression((coef, var_map[vid]))
                        for vid, coef in self.linear.items()
                        if coef
                    ]
                )
        if self.constant:
            ans += self.constant
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
        if not mult:
            # 0 * other, so there is nothing to add/change about
            # self.  We can just exit now.
            return
        if other.constant:
            self.constant += mult * other.constant
        if other.linear:
            _merge_dict(self.linear, mult, other.linear)
        if other.quadratic:
            if not self.quadratic:
                self.quadratic = {}
            _merge_dict(self.quadratic, mult, other.quadratic)
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
def _mul_linear_linear(varOrder, linear1, linear2):
    quadratic = {}
    for vid1, coef1 in linear1.items():
        for vid2, coef2 in linear2.items():
            if varOrder(vid1) < varOrder(vid2):
                key = vid1, vid2
            else:
                key = vid2, vid1
            if key in quadratic:
                quadratic[key] += coef1 * coef2
            else:
                quadratic[key] = coef1 * coef2
    return quadratic


def _handle_product_linear_linear(visitor, node, arg1, arg2):
    _, arg1 = arg1
    _, arg2 = arg2
    # Quadratic first, because we will update linear in a minute
    arg1.quadratic = _mul_linear_linear(
        visitor.var_order.__getitem__, arg1.linear, arg2.linear
    )
    # Linear second, as this relies on knowing the original constants
    if not arg2.constant:
        arg1.linear = {}
    elif arg2.constant != 1:
        c = arg2.constant
        _linear = arg1.linear
        for vid, coef in _linear.items():
            _linear[vid] = c * coef
    if arg1.constant:
        _merge_dict(arg1.linear, arg1.constant, arg2.linear)
    # Finally, the constant and multipliers
    arg1.constant *= arg2.constant
    arg1.multiplier *= arg2.multiplier
    return _QUADRATIC, arg1


def _handle_product_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    if not visitor.expand_nonlinear_products:
        ans.nonlinear = to_expression(visitor, arg1) * to_expression(visitor, arg2)
        return _GENERAL, ans

    # We are multiplying (A + Bx + Cx^2 + D(x)) * (A + Bx + Cx^2 + Dx))
    _, x1 = arg1
    _, x2 = arg2
    ans.multiplier = x1.multiplier * x2.multiplier
    x1.multiplier = x2.multiplier = 1
    # x1.const * x2.const [AA]
    ans.constant = x1.constant * x2.constant
    # linear & quadratic terms
    if x2.constant:
        # [BA], [CA]
        c = x2.constant
        if c == 1:
            ans.linear = dict(x1.linear)
            if x1.quadratic:
                ans.quadratic = dict(x1.quadratic)
        else:
            ans.linear = {vid: c * coef for vid, coef in x1.linear.items()}
            if x1.quadratic:
                ans.quadratic = {k: c * coef for k, coef in x1.quadratic.items()}
    if x1.constant:
        # [AB]
        _merge_dict(ans.linear, x1.constant, x2.linear)
        # [AC]
        if x2.quadratic:
            if ans.quadratic:
                _merge_dict(ans.quadratic, x1.constant, x2.quadratic)
            elif x1.constant == 1:
                ans.quadratic = dict(x2.quadratic)
            else:
                c = x1.constant
                ans.quadratic = {k: c * coef for k, coef in x2.quadratic.items()}
    # [BB]
    if x1.linear and x2.linear:
        quad = _mul_linear_linear(visitor.var_order.__getitem__, x1.linear, x2.linear)
        if ans.quadratic:
            _merge_dict(ans.quadratic, 1, quad)
        else:
            ans.quadratic = quad
    # [DA] + [DB] + [DC] + [DD]
    ans.nonlinear = 0
    if x1.nonlinear is not None:
        ans.nonlinear += x1.nonlinear * x2.to_expression(visitor)
    x1.nonlinear = None
    x2.constant = 0
    x1_c = x1.constant
    x1.constant = 0
    x1_lin = x1.linear
    x1.linear = {}
    # [CB] + [CC] + [CD]
    if x1.quadratic:
        ans.nonlinear += x1.to_expression(visitor) * x2.to_expression(visitor)
        x1.quadratic = None
    x2.linear = {}
    # [BC] + [BD]
    if x1_lin and (x2.nonlinear is not None or x2.quadratic):
        x1.linear = x1_lin
        ans.nonlinear += x1.to_expression(visitor) * x2.to_expression(visitor)
    # [AD]
    if x1_c and x2.nonlinear is not None:
        ans.nonlinear += x1_c * x2.nonlinear
    return _GENERAL, ans


_exit_node_handlers[ProductExpression].update(
    {
        (_CONSTANT, _QUADRATIC): linear._handle_product_constant_ANY,
        (_LINEAR, _QUADRATIC): _handle_product_nonlinear,
        (_QUADRATIC, _QUADRATIC): _handle_product_nonlinear,
        (_GENERAL, _QUADRATIC): _handle_product_nonlinear,
        (_QUADRATIC, _CONSTANT): linear._handle_product_ANY_constant,
        (_QUADRATIC, _LINEAR): _handle_product_nonlinear,
        (_QUADRATIC, _GENERAL): _handle_product_nonlinear,
        # Replace handler from the linear walker
        (_LINEAR, _LINEAR): _handle_product_linear_linear,
        (_GENERAL, _GENERAL): _handle_product_nonlinear,
        (_GENERAL, _LINEAR): _handle_product_nonlinear,
        (_LINEAR, _GENERAL): _handle_product_nonlinear,
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
_exit_node_handlers[Expression][(_QUADRATIC,)] = linear._handle_named_ANY

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

#
# RELATIONAL handlers
#
_exit_node_handlers[EqualityExpression].update(
    {
        (_CONSTANT, _QUADRATIC): linear._handle_equality_general,
        (_LINEAR, _QUADRATIC): linear._handle_equality_general,
        (_QUADRATIC, _QUADRATIC): linear._handle_equality_general,
        (_GENERAL, _QUADRATIC): linear._handle_equality_general,
        (_QUADRATIC, _CONSTANT): linear._handle_equality_general,
        (_QUADRATIC, _LINEAR): linear._handle_equality_general,
        (_QUADRATIC, _GENERAL): linear._handle_equality_general,
    }
)
_exit_node_handlers[InequalityExpression].update(
    {
        (_CONSTANT, _QUADRATIC): linear._handle_inequality_general,
        (_LINEAR, _QUADRATIC): linear._handle_inequality_general,
        (_QUADRATIC, _QUADRATIC): linear._handle_inequality_general,
        (_GENERAL, _QUADRATIC): linear._handle_inequality_general,
        (_QUADRATIC, _CONSTANT): linear._handle_inequality_general,
        (_QUADRATIC, _LINEAR): linear._handle_inequality_general,
        (_QUADRATIC, _GENERAL): linear._handle_inequality_general,
    }
)
_exit_node_handlers[RangedExpression].update(
    {
        (_CONSTANT, _QUADRATIC): linear._handle_ranged_general,
        (_LINEAR, _QUADRATIC): linear._handle_ranged_general,
        (_QUADRATIC, _QUADRATIC): linear._handle_ranged_general,
        (_GENERAL, _QUADRATIC): linear._handle_ranged_general,
        (_QUADRATIC, _CONSTANT): linear._handle_ranged_general,
        (_QUADRATIC, _LINEAR): linear._handle_ranged_general,
        (_QUADRATIC, _GENERAL): linear._handle_ranged_general,
    }
)


class QuadraticRepnVisitor(linear.LinearRepnVisitor):
    Result = QuadraticRepn
    exit_node_handlers = _exit_node_handlers
    exit_node_dispatcher = linear.ExitNodeDispatcher(
        linear._initialize_exit_node_dispatcher(_exit_node_handlers)
    )
    max_exponential_expansion = 2
