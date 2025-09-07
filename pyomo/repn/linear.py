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

import sys
from operator import itemgetter

from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import (
    native_types,
    native_numeric_types,
    native_complex_types,
)
from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    AbsExpression,
    UnaryFunctionExpression,
    Expr_ifExpression,
    MonomialTermExpression,
    LinearExpression,
    SumExpression,
    ExternalFunctionExpression,
    mutable_expression,
)
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.expr import is_fixed, value
from pyomo.core.base.expression import Expression
import pyomo.core.kernel as kernel
from pyomo.repn.util import (
    BeforeChildDispatcher,
    ExitNodeDispatcher,
    ExprType,
    FileDeterminism,
    FileDeterminism_to_SortComponents,
    InvalidNumber,
    OrderedVarRecorder,
    VarRecorder,
    apply_node_operation,
    complex_number_error,
    initialize_exit_node_dispatcher,
    nan,
    sum_like_expression_types,
    val2str,
)

_CONSTANT = ExprType.CONSTANT
_FIXED = ExprType.FIXED
_VARIABLE = ExprType.VARIABLE
_LINEAR = ExprType.LINEAR
_GENERAL = ExprType.GENERAL


def _merge_dict(dest_dict, src_dict, mult, flag):
    if not src_dict:
        return
    if flag == 1:
        for vid, coef in src_dict.items():
            if vid in dest_dict:
                dest_dict[vid] += coef
            else:
                dest_dict[vid] = coef
    elif not flag:
        # mult is 0.  There is nothing to do, unless the src_dict has an InvalidNumber
        for vid, coef in src_dict.items():
            if coef.__class__ is InvalidNumber:
                if vid in dest_dict:
                    dest_dict[vid] += mult * coef
                else:
                    dest_dict[vid] = mult * coef
    else:
        for vid, coef in src_dict.items():
            if vid in dest_dict:
                dest_dict[vid] += mult * coef
            else:
                dest_dict[vid] = mult * coef


class LinearRepn(object):
    __slots__ = ("multiplier", "constant", "linear", "nonlinear")

    def __init__(self):
        self.multiplier = 1
        self.constant = 0
        self.linear = {}
        self.nonlinear = None

    def __str__(self):
        linear = (
            "{"
            + ", ".join(f"{val2str(k)}: {val2str(v)}" for k, v in self.linear.items())
            + "}"
        )
        return (
            f"{self.__class__.__name__}(mult={val2str(self.multiplier)}, "
            f"const={val2str(self.constant)}, "
            f"linear={linear}, "
            f"nonlinear={self.nonlinear})"
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
        elif self.linear:
            return _LINEAR, self
        else:
            return _CONSTANT, self.multiplier * self.constant

    def duplicate(self):
        ans = self.__class__.__new__(self.__class__)
        ans.multiplier = self.multiplier
        ans.constant = self.constant
        ans.linear = dict(self.linear)
        ans.nonlinear = self.nonlinear
        return ans

    def to_expression(self, visitor):
        if self.nonlinear is not None:
            # We want to start with the nonlinear term (and use
            # assignment to ans instead of addition) in case the term is
            # a non-numeric node (like a relational expression)
            ans = self.nonlinear
        else:
            ans = 0
        with mutable_expression() as e:
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
        so that we can directly use a LinearRepn() as a `data` object in
        the expression walker (thereby allowing us to use the default
        implementation of acceptChildResult [which calls
        `data.append()`] and avoid the function call for a custom
        callback).

        """
        # Note that self.multiplier will always be 1 (we only call append()
        # within a sum, so there is no opportunity for self.multiplier to
        # change). Omitting the assertion for efficiency.
        # assert self.multiplier == 1
        _type, other = other
        if _type <= _FIXED:  # Note: catching _FIXED and _CONSTANT
            self.constant += other
            return

        other_mult_flag = self.multiplier_flag(other.multiplier)
        if other_mult_flag == 1:
            self.constant += other.constant
            if other.linear:
                for vid, coef in other.linear.items():
                    if vid in self.linear:
                        self.linear[vid] += coef
                    else:
                        self.linear[vid] = coef
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
        else:
            #
            # mult != 0 or 1
            #
            if self.constant_flag(other.constant):
                self.constant += mult * other.constant
            if other.linear:
                for vid, coef in other.linear.items():
                    if vid in self.linear:
                        self.linear[vid] += mult * coef
                    else:
                        self.linear[vid] = mult * coef
        if other.nonlinear is not None:
            if self.nonlinear is None:
                self.nonlinear = mult * other.nonlinear
            else:
                self.nonlinear += mult * other.nonlinear


def to_expression(visitor, arg):
    if arg[0] <= _VARIABLE:  # Note: catching _VARIABLE, _FIXED, and _CONSTANT
        return arg[1]
    else:
        return arg[1].to_expression(visitor)


#
# NEGATION handlers
#


def _handle_negation_constant(visitor, node, arg):
    return (_CONSTANT, -1 * arg[1])


def _handle_negation_fixed(visitor, node, arg):
    return (_FIXED, -1 * arg[1])


def _handle_negation_ANY(visitor, node, arg):
    arg[1].multiplier *= -1
    return arg


#
# PRODUCT handlers
#


def _handle_product_constant_constant(visitor, node, arg1, arg2):
    ans = arg1[1] * arg2[1]
    if ans.__class__ is InvalidNumber:
        constant_flag = visitor.Result.constant_flag
        if not constant_flag(arg1[1]) or not constant_flag(arg2[1]):
            a = val2str(arg1[1])
            b = val2str(arg2[1])
            deprecation_warning(
                f"Encountered {a}*{b} in expression tree.  "
                "Mapping the NaN result to 0 for compatibility "
                "with the lp_v1 writer.  In the future, this NaN "
                "will be preserved/emitted to comply with IEEE-754.",
                version='6.6.0',
            )
            return _CONSTANT, 0
    return _CONSTANT, ans


def _handle_product_fixed_fixed(visitor, node, arg1, arg2):
    # This is valid for fixed * constant, and fixed * fixed
    return _FIXED, arg1[1] * arg2[1]


def _handle_product_constant_ANY(visitor, node, arg1, arg2):
    arg2[1].multiplier *= arg1[1]
    return arg2


def _handle_product_ANY_constant(visitor, node, arg1, arg2):
    arg1[1].multiplier *= arg2[1]
    return arg1


def _handle_product_nonlinear(visitor, node, arg1, arg2):
    # Note: the expectation is that this method is not called often: the
    # Linear visitor is generally expected to be called on linear
    # expressions.  As such, we will not overly concern ourselves with
    # performance here.
    ans = visitor.Result()
    if not visitor.expand_nonlinear_products:
        ans.nonlinear = to_expression(visitor, arg1) * to_expression(visitor, arg2)
        return _GENERAL, ans
    #
    # We are multiplying (and expanding) m(A + Bx + C(x)) * m(A + Bx + C(x))
    _, x1 = arg1
    _, x2 = arg2
    # [mm]
    ans.multiplier = x1.multiplier * x2.multiplier
    # reset the multipliers so that to_expression doesn't re-apply them below
    x1.multiplier = x2.multiplier = 1
    # x1.const * x2.const [AA]
    x1_const_flag = ans.constant_flag(x1.constant)
    x2_const_flag = ans.constant_flag(x2.constant)
    if x1_const_flag and x2_const_flag:
        ans.constant = x1.constant * x2.constant
    # x1.linear * x2.const [BA] + x1.const * x2.linear [AB]
    _merge_dict(ans.linear, x1.linear, x2.constant, x2_const_flag)
    _merge_dict(ans.linear, x2.linear, x1.constant, x1_const_flag)
    NL = 0
    if x2.nonlinear is not None and (
        x1_const_flag or x2.nonlinear.__class__ is InvalidNumber
    ):
        # [AC]
        NL += x1.constant * x2.nonlinear
    if x1.nonlinear is not None and (
        x2_const_flag or x2.nonlinear.__class__ is InvalidNumber
    ):
        # [CA]
        NL += x2.constant * x1.nonlinear
    # [BB] + [BC] + [CB] + [CC]
    x1.constant = 0
    x2.constant = 0
    NL += to_expression(visitor, arg1) * to_expression(visitor, arg2)
    if NL.__class__ in sum_like_expression_types and NL.nargs() == 1:
        NL = NL.arg(0)
    ans.nonlinear = NL
    return _GENERAL, ans


#
# DIVISION handlers
#


def _handle_division_constant_constant(visitor, node, arg1, arg2):
    return _CONSTANT, apply_node_operation(node, (arg1[1], arg2[1]))


def _handle_division_fixed_fixed(visitor, node, arg1, arg2):
    return _FIXED, arg1[1] / arg2[1]


def _handle_division_ANY_constant(visitor, node, arg1, arg2):
    repn = arg1[1]
    # We can only apply the division operation (and reduce the
    # multiplier to a native value) if both the multiplier is a native
    # value AND the divisor is a constant.  We know the latter is true
    # here, but must check the former.  There is also a special case if
    # the divisor is 0: then we can reduce the multiplier to an
    # InvalidNumber (using apply_node_operation) regardless of what the
    # dividend is.  Again, note that arg2 is a constant, so we can check
    # it for 0 with bool()
    if repn.multiplier.__class__ in native_numeric_types or not arg2[1]:
        repn.multiplier = apply_node_operation(node, (repn.multiplier, arg2[1]))
    else:
        repn.multiplier /= arg2[1]
    return arg1


def _handle_division_ANY_fixed(visitor, node, arg1, arg2):
    arg1[1].multiplier /= arg2[1]
    return arg1


def _handle_division_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    ans.nonlinear = to_expression(visitor, arg1) / to_expression(visitor, arg2)
    return _GENERAL, ans


#
# EXPONENTIATION handlers
#


def _handle_pow_constant_constant(visitor, node, arg1, arg2):
    ans = apply_node_operation(node, (arg1[1], arg2[1]))
    if ans.__class__ in native_complex_types:
        ans = complex_number_error(ans, visitor, node)
    return _CONSTANT, ans


def _handle_pow_fixed_fixed(visitor, node, arg1, arg2):
    return _FIXED, arg1[1] ** arg2[1]


def _handle_pow_ANY_constant(visitor, node, arg1, arg2):
    _, exp = arg2
    if exp == 1:
        return arg1
    elif exp > 1 and exp <= visitor.max_exponential_expansion and int(exp) == exp:
        _type, _arg = arg1
        ans = _type, _arg.duplicate()
        for i in range(1, int(exp)):
            ans = visitor.exit_node_dispatcher[(ProductExpression, ans[0], _type)](
                visitor, None, ans, (_type, _arg.duplicate())
            )
        return ans
    elif not exp:
        return _CONSTANT, 1
    return _handle_pow_nonlinear(visitor, node, arg1, arg2)


def _handle_pow_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    ans.nonlinear = to_expression(visitor, arg1) ** to_expression(visitor, arg2)
    return _GENERAL, ans


#
# ABS and UNARY handlers
#


def _handle_unary_constant(visitor, node, arg):
    ans = apply_node_operation(node, (arg[1],))
    # Unary includes sqrt() which can return complex numbers
    if ans.__class__ in native_complex_types:
        ans = complex_number_error(ans, visitor, node)
    return _CONSTANT, ans


def _handle_unary_fixed(visitor, node, arg):
    return _FIXED, node.create_node_with_local_data((arg[1],))


def _handle_unary_nonlinear(visitor, node, arg):
    ans = visitor.Result()
    ans.nonlinear = node.create_node_with_local_data((to_expression(visitor, arg),))
    return _GENERAL, ans


#
# NAMED EXPRESSION handlers
#


def _handle_named_constant(visitor, node, arg1):
    # Record this common expression
    visitor.subexpression_cache[id(node)] = arg1
    return arg1


def _handle_named_ANY(visitor, node, arg1):
    # Record this common expression
    visitor.subexpression_cache[id(node)] = arg1
    _type, arg1 = arg1
    return _type, arg1.duplicate()


#
# EXPR_IF handlers
#


def _handle_expr_if_const(visitor, node, arg1, arg2, arg3):
    _type, _test = arg1
    assert _type is _CONSTANT
    if _test:
        if _test.__class__ is InvalidNumber:
            # nan
            return _handle_expr_if_nonlinear(visitor, node, arg1, arg2, arg3)
        return arg2
    else:
        return arg3


def _handle_expr_if_nonlinear(visitor, node, arg1, arg2, arg3):
    # Note: guaranteed that arg1 is not _CONSTANT
    ans = visitor.Result()
    ans.nonlinear = Expr_ifExpression(
        (
            to_expression(visitor, arg1),
            to_expression(visitor, arg2),
            to_expression(visitor, arg3),
        )
    )
    return _GENERAL, ans


#
# Relational expression handlers
#


def _handle_equality_const(visitor, node, arg1, arg2):
    # It is exceptionally likely that if we get here, one of the
    # arguments is an InvalidNumber
    args, causes = InvalidNumber.parse_args(arg1[1], arg2[1])
    try:
        ans = args[0] == args[1]
    except:
        ans = False
        causes.append(str(sys.exc_info()[1]))
    if causes:
        ans = InvalidNumber(ans, causes)
    return _CONSTANT, ans


def _handle_equality_general(visitor, node, arg1, arg2):
    ans = visitor.Result()
    ans.nonlinear = EqualityExpression(
        (to_expression(visitor, arg1), to_expression(visitor, arg2))
    )
    return _GENERAL, ans


def _handle_inequality_const(visitor, node, arg1, arg2):
    # It is exceptionally likely that if we get here, one of the
    # arguments is an InvalidNumber
    args, causes = InvalidNumber.parse_args(arg1[1], arg2[1])
    try:
        ans = args[0] <= args[1]
    except:
        ans = False
        causes.append(str(sys.exc_info()[1]))
    if causes:
        ans = InvalidNumber(ans, causes)
    return _CONSTANT, ans


def _handle_inequality_general(visitor, node, arg1, arg2):
    ans = visitor.Result()
    ans.nonlinear = InequalityExpression(
        (to_expression(visitor, arg1), to_expression(visitor, arg2)), node.strict
    )
    return _GENERAL, ans


def _handle_ranged_const(visitor, node, arg1, arg2, arg3):
    # It is exceptionally likely that if we get here, one of the
    # arguments is an InvalidNumber
    args, causes = InvalidNumber.parse_args(arg1[1], arg2[1], arg3[1])
    try:
        ans = args[0] <= args[1] <= args[2]
    except:
        ans = False
        causes.append(str(sys.exc_info()[1]))
    if causes:
        ans = InvalidNumber(ans, causes)
    return _CONSTANT, ans


def _handle_ranged_general(visitor, node, arg1, arg2, arg3):
    ans = visitor.Result()
    ans.nonlinear = RangedExpression(
        (
            to_expression(visitor, arg1),
            to_expression(visitor, arg2),
            to_expression(visitor, arg3),
        ),
        node.strict,
    )
    return _GENERAL, ans


def define_exit_node_handlers(_exit_node_handlers=None):
    if _exit_node_handlers is None:
        _exit_node_handlers = {}
    _exit_node_handlers[NegationExpression] = {
        None: _handle_negation_ANY,
        (_CONSTANT,): _handle_negation_constant,
        (_FIXED,): _handle_negation_fixed,
    }
    _exit_node_handlers[ProductExpression] = {
        None: _handle_product_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_product_constant_constant,
        (_CONSTANT, _FIXED): _handle_product_fixed_fixed,
        (_CONSTANT, _LINEAR): _handle_product_constant_ANY,
        (_CONSTANT, _GENERAL): _handle_product_constant_ANY,
        (_FIXED, _CONSTANT): _handle_product_fixed_fixed,
        (_FIXED, _FIXED): _handle_product_fixed_fixed,
        (_FIXED, _LINEAR): _handle_product_constant_ANY,
        (_FIXED, _GENERAL): _handle_product_constant_ANY,
        (_LINEAR, _CONSTANT): _handle_product_ANY_constant,
        (_LINEAR, _FIXED): _handle_product_ANY_constant,
        (_GENERAL, _CONSTANT): _handle_product_ANY_constant,
        (_GENERAL, _FIXED): _handle_product_ANY_constant,
    }
    _exit_node_handlers[MonomialTermExpression] = _exit_node_handlers[ProductExpression]
    _exit_node_handlers[DivisionExpression] = {
        None: _handle_division_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_division_constant_constant,
        (_CONSTANT, _FIXED): _handle_division_fixed_fixed,
        (_FIXED, _CONSTANT): _handle_division_fixed_fixed,
        (_FIXED, _FIXED): _handle_division_fixed_fixed,
        (_LINEAR, _CONSTANT): _handle_division_ANY_constant,
        (_LINEAR, _FIXED): _handle_division_ANY_fixed,
        (_GENERAL, _CONSTANT): _handle_division_ANY_constant,
        (_GENERAL, _FIXED): _handle_division_ANY_fixed,
    }
    _exit_node_handlers[PowExpression] = {
        None: _handle_pow_nonlinear,
        (_CONSTANT, _CONSTANT): _handle_pow_constant_constant,
        (_CONSTANT, _FIXED): _handle_pow_fixed_fixed,
        (_FIXED, _CONSTANT): _handle_pow_fixed_fixed,
        (_FIXED, _FIXED): _handle_pow_fixed_fixed,
        (_LINEAR, _CONSTANT): _handle_pow_ANY_constant,
        (_GENERAL, _CONSTANT): _handle_pow_ANY_constant,
    }
    _exit_node_handlers[UnaryFunctionExpression] = {
        None: _handle_unary_nonlinear,
        (_CONSTANT,): _handle_unary_constant,
        (_FIXED,): _handle_unary_fixed,
    }
    _exit_node_handlers[AbsExpression] = _exit_node_handlers[UnaryFunctionExpression]
    _exit_node_handlers[Expression] = {
        None: _handle_named_ANY,
        (_CONSTANT,): _handle_named_constant,
        (_FIXED,): _handle_named_constant,
    }
    _exit_node_handlers[Expr_ifExpression] = {None: _handle_expr_if_nonlinear}
    for j in (_CONSTANT, _LINEAR, _GENERAL):
        for k in (_CONSTANT, _LINEAR, _GENERAL):
            _exit_node_handlers[Expr_ifExpression][
                _CONSTANT, j, k
            ] = _handle_expr_if_const
    _exit_node_handlers[EqualityExpression] = {
        None: _handle_equality_general,
        (_CONSTANT, _CONSTANT): _handle_equality_const,
    }
    _exit_node_handlers[InequalityExpression] = {
        None: _handle_inequality_general,
        (_CONSTANT, _CONSTANT): _handle_inequality_const,
    }
    _exit_node_handlers[RangedExpression] = {
        None: _handle_ranged_general,
        (_CONSTANT, _CONSTANT, _CONSTANT): _handle_ranged_const,
    }
    return _exit_node_handlers


class LinearBeforeChildDispatcher(BeforeChildDispatcher):
    def __init__(self):
        # Special handling for external functions: will be handled
        # as terminal nodes from the point of view of the visitor
        self[ExternalFunctionExpression] = self._before_external
        # Special linear / summation expressions
        self[MonomialTermExpression] = self._before_monomial
        self[LinearExpression] = self._before_linear
        self[SumExpression] = self._before_general_expression

    @staticmethod
    def _before_var(visitor, child):
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                return False, (_CONSTANT, visitor.check_constant(child.value, child))
            visitor.var_recorder.add(child)
        ans = visitor.Result()
        ans.linear[_id] = 1
        return False, (_LINEAR, ans)

    @staticmethod
    def _before_monomial(visitor, child):
        #
        # The following are performance optimizations for common
        # situations (Monomial terms and Linear expressions)
        #
        arg1, arg2 = child._args_
        if arg1.__class__ not in native_types:
            try:
                arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
            except (ValueError, ArithmeticError):
                return True, None

        # We want to check / update the var_map before processing "0"
        # coefficients so that we are consistent with what gets added to the
        # var_map (e.g., 0*x*y: y is processed by _before_var and will
        # always be added, but x is processed here)
        _id = id(arg2)
        if _id not in visitor.var_map:
            if arg2.fixed:
                return False, (
                    _CONSTANT,
                    arg1 * visitor.check_constant(arg2.value, arg2),
                )
            visitor.var_recorder.add(arg2)

        # Trap multiplication by 0 and nan.  Note that arg1 was reduced
        # to a numeric value at the beginning of this method.
        if not arg1:
            if arg2.fixed:
                arg2 = visitor.check_constant(arg2.value, arg2)
                if arg2.__class__ is InvalidNumber:
                    deprecation_warning(
                        f"Encountered {arg1}*{val2str(arg2)} in expression "
                        "tree.  Mapping the NaN result to 0 for compatibility "
                        "with the lp_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.6.0',
                    )
            return False, (_CONSTANT, arg1)

        ans = visitor.Result()
        ans.linear[_id] = arg1
        return False, (_LINEAR, ans)

    @staticmethod
    def _before_linear(visitor, child):
        var_map = visitor.var_map
        ans = visitor.Result()
        const = 0
        linear = ans.linear
        for arg in child.args:
            if arg.__class__ is MonomialTermExpression:
                arg1, arg2 = arg._args_
                if arg1.__class__ not in native_types:
                    try:
                        arg1 = visitor.check_constant(visitor.evaluate(arg1), arg1)
                    except (ValueError, ArithmeticError):
                        return True, None

                # Trap multiplication by 0 and nan.  Note that arg1 was
                # reduced to a numeric value at the beginning of this
                # method.
                if not arg1:
                    if arg2.fixed:
                        arg2 = visitor.check_constant(arg2.value, arg2)
                        if arg2.__class__ is InvalidNumber:
                            deprecation_warning(
                                f"Encountered {arg1}*{val2str(arg2)} in expression "
                                "tree.  Mapping the NaN result to 0 for compatibility "
                                "with the lp_v1 writer.  In the future, this NaN "
                                "will be preserved/emitted to comply with IEEE-754.",
                                version='6.6.0',
                            )
                    continue

                _id = id(arg2)
                if _id not in var_map:
                    if arg2.fixed:
                        const += arg1 * visitor.check_constant(arg2.value, arg2)
                        continue
                    visitor.var_recorder.add(arg2)
                    linear[_id] = arg1
                elif _id in linear:
                    linear[_id] += arg1
                else:
                    linear[_id] = arg1
            elif arg.__class__ in native_numeric_types:
                const += arg
            elif arg.is_variable_type():
                _id = id(arg)
                if _id not in var_map:
                    if arg.fixed:
                        const += visitor.check_constant(arg.value, arg)
                        continue
                    visitor.var_recorder.add(arg)
                    linear[_id] = 1
                elif _id in linear:
                    linear[_id] += 1
                else:
                    linear[_id] = 1
            else:
                try:
                    const += visitor.check_constant(visitor.evaluate(arg), arg)
                except (ValueError, ArithmeticError):
                    return True, None
        if linear:
            ans.constant = const
            return False, (_LINEAR, ans)
        else:
            return False, (_CONSTANT, const)

    @staticmethod
    def _before_named_expression(visitor, child):
        _id = id(child)
        if _id in visitor.subexpression_cache:
            _type, expr = visitor.subexpression_cache[_id]
            if _type is _CONSTANT:
                return False, (_type, expr)
            else:
                return False, (_type, expr.duplicate())
        else:
            return True, None

    @staticmethod
    def _before_external(visitor, child):
        ans = visitor.Result()
        if all(is_fixed(arg) for arg in child.args):
            try:
                ans.constant = visitor.check_constant(visitor.evaluate(child), child)
                return False, (_CONSTANT, ans)
            except:
                pass
        ans.nonlinear = child
        return False, (_GENERAL, ans)


class LinearRepnVisitor(StreamBasedExpressionVisitor):
    Result = LinearRepn
    before_child_dispatcher = LinearBeforeChildDispatcher()
    exit_node_dispatcher = ExitNodeDispatcher(
        initialize_exit_node_dispatcher(define_exit_node_handlers())
    )
    expand_nonlinear_products = False
    max_exponential_expansion = 1

    def __init__(
        self,
        subexpression_cache,
        var_map=None,
        var_order=None,
        sorter=None,
        var_recorder=None,
    ):
        super().__init__()
        self.subexpression_cache = subexpression_cache
        if any(_ is not None for _ in (var_map, var_order, sorter)):
            if var_recorder is not None:
                raise ValueError(
                    "LinearRepnVisitor: cannot specify any of var_map, "
                    "var_order, or sorter with var_recorder"
                )
            deprecation_warning(
                "var_map, var_order, and sorter are deprecated arguments to "
                "LinearRepnVisitor().  Please pass the VarRecorder object directly.",
                version='6.8.1',
            )
            var_recorder = OrderedVarRecorder(var_map, var_order, sorter)
        if var_recorder is None:
            var_recorder = VarRecorder(
                {}, FileDeterminism_to_SortComponents(FileDeterminism.ORDERED)
            )
        self.var_recorder = var_recorder
        self.var_map = var_recorder.var_map
        self._eval_expr_visitor = _EvaluationVisitor(True)
        self.evaluate = self._eval_expr_visitor.dfs_postorder_stack

    def check_constant(self, ans, obj):
        if ans.__class__ not in native_numeric_types:
            # None can be returned from uninitialized Var/Param objects
            if ans is None:
                return InvalidNumber(
                    None, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                )
            if ans.__class__ is InvalidNumber:
                return ans
            elif ans.__class__ in native_complex_types:
                return complex_number_error(ans, self, obj)
            else:
                # It is possible to get other non-numeric types.  Most
                # common are bool and 1-element numpy.array().  We will
                # attempt to convert the value to a float before
                # proceeding.
                #
                # Note that as of NumPy 1.25, blindly casting a
                # 1-element ndarray to a float will generate a
                # deprecation warning.  We will explicitly test for
                # that, but want to do the test without triggering the
                # numpy import
                for cls in ans.__class__.__mro__:
                    if cls.__name__ == 'ndarray' and cls.__module__ == 'numpy':
                        if len(ans) == 1:
                            ans = ans[0]
                        break
                # TODO: we should check bool and warn/error (while bool is
                # convertible to float in Python, they have very
                # different semantic meanings in Pyomo).
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(
                        ans, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
                    )
        if ans != ans:
            return InvalidNumber(
                nan, f"'{obj}' evaluated to a nonnumeric value '{ans}'"
            )
        return ans

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        return self.before_child_dispatcher[child.__class__](self, child)

    def enterNode(self, node):
        # SumExpression are potentially large nary operators.  Directly
        # populate the result
        if node.__class__ in sum_like_expression_types:
            return node.args, self.Result()
        else:
            return node.args, []

    def exitNode(self, node, data):
        if data.__class__ is self.Result:
            return data.walker_exitNode()
        #
        # General expressions...
        #
        return self.exit_node_dispatcher[(node.__class__, *map(itemgetter(0), data))](
            self, node, *data
        )

    def finalizeResult(self, result):
        ans = result[1]
        if ans.__class__ is not self.Result:
            ans = self.Result()
            assert result[0] <= _FIXED  # Note: allowing _FIXED or _CONSTANT
            ans.constant = result[1]
            return ans

        mult_flag = ans.multiplier_flag(ans.multiplier)
        if mult_flag == 1:
            # mult is identity: only thing to do is filter out zero coefficients
            self._filter_zeros(ans)
            return ans
        elif not mult_flag:
            # the multiplier has cleared out the entire expression.
            # Warn if this is suppressing a NaN (unusual, and
            # non-standard, but we will wait to remove this behavior
            # for the time being)
            if ans.constant.__class__ is InvalidNumber or any(
                c.__class__ is InvalidNumber for c in ans.linear.values()
            ):
                deprecation_warning(
                    f"Encountered {ans.multiplier}*nan in expression tree.  "
                    "Mapping the NaN result to 0 for compatibility "
                    "with the lp_v1 writer.  In the future, this NaN "
                    "will be preserved/emitted to comply with IEEE-754.",
                    version='6.6.0',
                )
            return self.Result()

        # mult not in {0, 1}: factor it into the constant,
        # linear coefficients, and nonlinear term
        self._factor_multiplier_into_ans(ans, ans.multiplier)
        return ans

    def _filter_zeros(self, ans):
        _flag = ans.constant_flag
        # Note: creating the intermediate list is important, as we are
        # modifying the dict in place.
        for vid in [vid for vid, c in ans.linear.items() if not _flag(c)]:
            del ans.linear[vid]

    def _factor_multiplier_into_ans(self, ans, mult):
        _flag = ans.constant_flag
        linear = ans.linear
        zeros = []
        for vid, coef in linear.items():
            prod = mult * coef
            if _flag(prod):
                linear[vid] = prod
            else:
                zeros.append(vid)
        for vid in zeros:
            del linear[vid]
        if ans.nonlinear is not None:
            ans.nonlinear *= mult
        if _flag(ans.constant):
            ans.constant *= mult
        ans.multiplier = 1
