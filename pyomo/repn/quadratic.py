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

from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    Expr_ifExpression,
    mutable_expression,
)
import pyomo.repn.linear as linear
import pyomo.repn.util as util
from pyomo.repn.linear import _merge_dict, to_expression
from pyomo.repn.util import sum_like_expression_types, val2str, InvalidNumber

_CONSTANT = linear.ExprType.CONSTANT
_FIXED = linear.ExprType.FIXED
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
        linear = (
            "{"
            + ", ".join(f"{val2str(k)}: {val2str(v)}" for k, v in self.linear.items())
            + "}"
        )
        if self.quadratic is None:
            quadratic = None
        else:
            quadratic = (
                "{"
                + ", ".join(
                    f"({val2str(k1)}, {val2str(k2)}): {val2str(v)}"
                    for (k1, k2), v in self.quadratic.items()
                )
                + "}"
            )
        return (
            f"{self.__class__.__name__}(mult={val2str(self.multiplier)}, "
            f"const={val2str(self.constant)}, "
            f"linear={linear}, quadratic={quadratic}, "
            f"nonlinear={val2str(self.nonlinear)})"
        )

    def __repr__(self):
        return str(self)

    @staticmethod
    def constant_flag(val):
        return val

    @staticmethod
    def multiplier_flag(val):
        return val

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
        with mutable_expression() as e:
            if self.quadratic:
                for (x1, x2), coef in self.quadratic.items():
                    if self.multiplier_flag(coef):
                        if x1 == x2:
                            e += coef * var_map[x1] ** 2
                        else:
                            e += coef * (var_map[x1] * var_map[x2])
            if self.linear:
                var_map = visitor.var_map
                for vid, coef in self.linear.items():
                    if self.multiplier_flag(coef):
                        e += coef * var_map[vid]
            if self.constant_flag(self.constant):
                e += self.constant
        if e.nargs() > 1:
            ans += e
        elif e.nargs() == 1:
            ans += e.arg(0)
        if self.multiplier_flag(self.multiplier) != 1:
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
        if _type <= _FIXED:
            self.constant += other
            return

        other_mult_flag = self.multiplier_flag(other.multiplier)
        if other_mult_flag == 1:
            self.constant += other.constant
            if other.linear:
                l = self.linear
                for vid, coef in other.linear.items():
                    if vid in l:
                        l[vid] += coef
                    else:
                        l[vid] = coef
            if other.quadratic:
                if not self.quadratic:
                    self.quadratic = other.quadratic
                else:
                    q = self.quadratic
                    for vid, coef in other.quadratic.items():
                        if vid in q:
                            q[vid] += coef
                        else:
                            q[vid] = coef
            if other.nonlinear is not None:
                if self.nonlinear is None:
                    self.nonlinear = other.nonlinear
                else:
                    self.nonlinear += other.nonlinear
            return

        mult = other.multiplier
        if not other_mult_flag:
            # 0 * other, so you would think that there is nothing to
            # add/change about self.  However, there is a chance
            # that other contains an InvalidNumber, so we should go
            # looking for it...
            if other.constant.__class__ is InvalidNumber:
                self.constant += mult * other.constant
            for vid, coef in other.linear.items():
                if coef.__class__ is InvalidNumber:
                    if vid in self.linear:
                        self.linear[vid] += mult * coef
                    else:
                        self.linear[vid] = mult * coef
            if other.quadratic:
                for vid, coef in other.quadratic.items():
                    if coef.__class__ is InvalidNumber:
                        if not self.quadratic:
                            self.quadratic = {}
                        if vid in self.quadratic:
                            self.quadratic[vid] += mult * coef
                        else:
                            self.quadratic[vid] = mult * coef
        else:
            #
            # other.multiplier other than 0 or 1:
            #
            if self.constant_flag(other.constant):
                self.constant += mult * other.constant
            if other.linear:
                l = self.linear
                for vid, coef in other.linear.items():
                    if vid in l:
                        l[vid] += mult * coef
                    else:
                        l[vid] = mult * coef
            if other.quadratic:
                if self.quadratic:
                    q = self.quadratic
                else:
                    q = self.quadratic = {}
                for vid, coef in other.quadratic.items():
                    if vid in q:
                        q[vid] += mult * coef
                    else:
                        q[vid] = mult * coef

        if other.nonlinear is not None:
            if self.nonlinear is None:
                self.nonlinear = mult * other.nonlinear
            else:
                self.nonlinear += mult * other.nonlinear


def _mul_linear_linear(visitor, linear1, linear2):
    vo = visitor.var_recorder.var_order
    quadratic = {}
    l2 = [(k, v, vo[k]) for k, v in linear2.items()]
    for vid1, coef1 in linear1.items():
        o1 = vo[vid1]
        for vid2, coef2, o2 in l2:
            if o1 < o2:
                key = vid1, vid2
            else:
                key = vid2, vid1
            if key in quadratic:
                quadratic[key] += coef1 * coef2
            else:
                quadratic[key] = coef1 * coef2
    return quadratic


def _handle_product_linear_linear(visitor, node, arg1, arg2):
    # We are multiplying (A + Bx) * (A + Bx)
    _, arg1 = arg1
    _, arg2 = arg2
    # [BB]: Quadratic first, because we will update linear in a minute
    arg1.quadratic = _mul_linear_linear(visitor, arg1.linear, arg2.linear)
    # Linear second, as this relies on knowing the original constants
    # [BA]
    arg1_const_flag = arg1.constant_flag(arg1.constant)
    arg2_const_flag = arg2.constant_flag(arg2.constant)
    if arg2_const_flag == 1:
        pass
    elif not arg2_const_flag:
        # linear * 0, so you would think there is nothing to do, but we
        # need to preserve any InvalidNumbers
        arg1.linear = {
            vid: coef * arg2.constant
            for key, coef in arg1.linear.items()
            if coef.__class__ is InvalidNumber
        }
    else:
        c = arg2.constant
        l = arg1.linear
        for vid, coef in l.items():
            l[vid] = c * coef
    # [AB]
    _merge_dict(arg1.linear, arg2.linear, arg1.constant, arg1_const_flag)
    # [AA]: Finally, the constant and multipliers
    if arg1_const_flag and arg2_const_flag:
        arg1.constant *= arg2.constant
    else:
        arg1.constant = 0
    arg1.multiplier *= arg2.multiplier
    return _QUADRATIC, arg1


def _handle_product_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    # We are multiplying (A + Bx + Cx^2 + D(x)) * (A + Bx + Cx^2 + Dx))
    _t1, x1 = arg1
    _t2, x2 = arg2
    if not visitor.expand_nonlinear_products:
        ans.nonlinear = to_expression(visitor, arg1) * to_expression(visitor, arg2)
        return _GENERAL, ans

    ans.multiplier = x1.multiplier * x2.multiplier
    x1.multiplier = x2.multiplier = 1
    # x1.const * x2.const [AA]
    ans.constant = x1.constant * x2.constant
    # linear & quadratic terms
    ans.quadratic = {}
    # [BA], [CA]
    x1_const_flag = ans.constant_flag(x1.constant)
    x2_const_flag = ans.constant_flag(x2.constant)
    _merge_dict(ans.linear, x1.linear, x2.constant, x2_const_flag)
    _merge_dict(ans.quadratic, x1.quadratic, x2.constant, x2_const_flag)
    # [AB], [AC]
    _merge_dict(ans.linear, x2.linear, x1.constant, x1_const_flag)
    _merge_dict(ans.quadratic, x2.quadratic, x1.constant, x1_const_flag)
    # [BB]
    if x1.linear and x2.linear:
        quad = _mul_linear_linear(visitor, x1.linear, x2.linear)
        if ans.quadratic:
            _merge_dict(ans.quadratic, quad, 1, 1)
        else:
            ans.quadratic = quad
    # [DA] + [DB] + [DC] + [DD]
    NL = 0
    if x1.nonlinear is not None:
        NL += x1.nonlinear * x2.to_expression(visitor)
    x1.nonlinear = None
    x2.constant = 0
    x1_c = x1.constant
    x1.constant = 0
    x1_lin = x1.linear
    x1.linear = {}
    # [CB] + [CC] + [CD]
    if x1.quadratic:
        NL += x1.to_expression(visitor) * x2.to_expression(visitor)
        x1.quadratic = None
    x2.linear = {}
    # [BC] + [BD]
    if x1_lin and (x2.nonlinear is not None or x2.quadratic):
        x1.linear = x1_lin
        NL += x1.to_expression(visitor) * x2.to_expression(visitor)
    # [AD]
    if x1_const_flag and x2.nonlinear is not None:
        NL += x1_c * x2.nonlinear
    if NL.__class__ in sum_like_expression_types and NL.nargs() == 1:
        NL = NL.arg(0)
    ans.nonlinear = NL
    return _GENERAL, ans


def _handle_pow_linear_constant(visitor, node, arg1, arg2):
    _, exp = arg2
    if exp == 2:
        _, ans = arg1
        ans.multiplier = ans.multiplier**2
        ans.quadratic = {(vid, vid): coef * coef for vid, coef in ans.linear.items()}
        if len(ans.linear) > 1:
            vo = visitor.var_recorder.var_order
            l = [(vid, coef, vo[vid]) for vid, coef in ans.linear.items()]
            for i, (x1, c1, o1) in enumerate(l):
                for j, (x2, c2, o2) in enumerate(l):
                    if i == j:
                        break
                    if o1 < o2:
                        ans.quadratic[x1, x2] = 2 * c1 * c2
                    else:
                        ans.quadratic[x2, x1] = 2 * c1 * c2
        if ans.constant_flag(ans.constant):
            c = 2 * ans.constant
            for vid in ans.linear:
                ans.linear[vid] *= c
            ans.constant = ans.constant**2
        else:
            ans.linear = {}
        return _QUADRATIC, ans
    return linear._handle_pow_ANY_constant(visitor, node, arg1, arg2)


def define_exit_node_handlers(_exit_node_handlers=None):
    if _exit_node_handlers is None:
        _exit_node_handlers = {}
    linear.define_exit_node_handlers(_exit_node_handlers)
    #
    # NEGATION
    #
    _exit_node_handlers[NegationExpression][(_QUADRATIC,)] = linear._handle_negation_ANY
    #
    # PRODUCT
    #
    _exit_node_handlers[ProductExpression].update(
        {
            None: _handle_product_nonlinear,
            (_CONSTANT, _QUADRATIC): linear._handle_product_constant_ANY,
            (_FIXED, _QUADRATIC): linear._handle_product_constant_ANY,
            (_QUADRATIC, _CONSTANT): linear._handle_product_ANY_constant,
            (_QUADRATIC, _FIXED): linear._handle_product_ANY_constant,
            # Replace handler from the linear walker
            (_LINEAR, _LINEAR): _handle_product_linear_linear,
        }
    )
    #
    # DIVISION
    #
    _exit_node_handlers[DivisionExpression].update(
        {
            (_QUADRATIC, _CONSTANT): linear._handle_division_ANY_constant,
            (_QUADRATIC, _FIXED): linear._handle_division_ANY_fixed,
        }
    )
    #
    # EXPONENTIATION
    #
    _exit_node_handlers[PowExpression].update(
        {
            (_LINEAR, _CONSTANT): _handle_pow_linear_constant,
            (_QUADRATIC, _CONSTANT): linear._handle_pow_ANY_constant,
        }
    )
    #
    # ABS and UNARY handlers
    #
    # (no changes needed)
    #
    # NAMED EXPRESSION handlers
    #
    # (no changes needed)
    #
    # EXPR_IF handlers
    #
    _exit_node_handlers[Expr_ifExpression].update(
        {
            (_CONSTANT, i, _QUADRATIC): linear._handle_expr_if_const
            for i in (_CONSTANT, _FIXED, _LINEAR, _QUADRATIC, _GENERAL)
        }
    )
    _exit_node_handlers[Expr_ifExpression].update(
        {
            (_CONSTANT, _QUADRATIC, i): linear._handle_expr_if_const
            for i in (_CONSTANT, _FIXED, _LINEAR, _GENERAL)
        }
    )
    #
    # RELATIONAL handlers
    #
    # (no changes needed)
    return _exit_node_handlers


class QuadraticRepnVisitor(linear.LinearRepnVisitor):
    Result = QuadraticRepn
    exit_node_dispatcher = linear.ExitNodeDispatcher(
        util.initialize_exit_node_dispatcher(define_exit_node_handlers())
    )
    max_exponential_expansion = 2

    def _filter_zeros(self, ans):
        _flag = ans.constant_flag
        # Note: creating the intermediate list is important, as we are
        # modifying the dict in place.
        for vid in [vid for vid, c in ans.linear.items() if not _flag(c)]:
            del ans.linear[vid]
        if ans.quadratic:
            for vid in [vid for vid, c in ans.quadratic.items() if not _flag(c)]:
                del ans.quadratic[vid]
            if not ans.quadratic:
                ans.quadratic = None

    def _factor_multiplier_into_ans(self, ans, mult):
        _flag = ans.constant_flag
        linear = ans.linear
        zeros = []
        for vid, coef in linear.items():
            prod = coef * mult
            if _flag(prod):
                linear[vid] = prod
            else:
                zeros.append(vid)
        for vid in zeros:
            del linear[vid]
        if ans.quadratic:
            quadratic = ans.quadratic
            zeros = []
            for vid, coef in quadratic.items():
                prod = coef * mult
                if _flag(prod):
                    quadratic[vid] = prod
                else:
                    zeros.append(vid)
            for vid in zeros:
                del quadratic[vid]
            if not quadratic:
                ans.quadratic = None
        if ans.nonlinear is not None:
            ans.nonlinear *= mult
        if _flag(ans.constant):
            ans.constant *= mult
        ans.multiplier = 1
