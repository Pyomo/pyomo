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
import logging
import sys
from operator import itemgetter
from itertools import filterfalse

from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import native_types, native_numeric_types
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
    NPV_SumExpression,
    ExternalFunctionExpression,
)
from pyomo.core.expr.relational_expr import (
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.expr import is_fixed, value
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.repn.util import (
    ExprType,
    InvalidNumber,
    apply_node_operation,
    complex_number_error,
    nan,
    sum_like_expression_types,
)

logger = logging.getLogger(__name__)

_CONSTANT = ExprType.CONSTANT
_LINEAR = ExprType.LINEAR
_GENERAL = ExprType.GENERAL


def _merge_dict(dest_dict, mult, src_dict):
    if mult == 1:
        for vid, coef in src_dict.items():
            if vid in dest_dict:
                dest_dict[vid] += coef
            else:
                dest_dict[vid] = coef
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
        return (
            f"LinearRepn(mult={self.multiplier}, const={self.constant}, "
            f"linear={self.linear}, nonlinear={self.nonlinear})"
        )

    def __repr__(self):
        return str(self)

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
            # assignment) in case the term is a non-numeric node (like a
            # relational expression)
            ans = self.nonlinear
        else:
            ans = 0
        if self.linear:
            var_map = visitor.var_map
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
        so that we can directly use a LinearRepn() as a data object in
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
        if other.constant:
            self.constant += mult * other.constant
        if other.linear:
            _merge_dict(self.linear, mult, other.linear)
        if other.nonlinear is not None:
            if mult != 1:
                nl = mult * other.nonlinear
            else:
                nl = other.nonlinear
            if self.nonlinear is None:
                self.nonlinear = nl
            else:
                self.nonlinear += nl


def to_expression(visitor, arg):
    if arg[0] is _CONSTANT:
        return arg[1]
    else:
        return arg[1].to_expression(visitor)


_exit_node_handlers = {}

#
# NEGATION handlers
#


def _handle_negation_constant(visitor, node, arg):
    return (_CONSTANT, -1 * arg[1])


def _handle_negation_ANY(visitor, node, arg):
    arg[1].multiplier *= -1
    return arg


_exit_node_handlers[NegationExpression] = {
    (_CONSTANT,): _handle_negation_constant,
    (_LINEAR,): _handle_negation_ANY,
    (_GENERAL,): _handle_negation_ANY,
}

#
# PRODUCT handlers
#


def _handle_product_constant_constant(visitor, node, arg1, arg2):
    _, arg1 = arg1
    _, arg2 = arg2
    ans = arg1 * arg2
    if ans != ans:
        if not arg1 or not arg2:
            deprecation_warning(
                f"Encountered {str(arg1)}*{str(arg2)} in expression tree.  "
                "Mapping the NaN result to 0 for compatibility "
                "with the lp_v1 writer.  In the future, this NaN "
                "will be preserved/emitted to comply with IEEE-754.",
                version='6.6.0',
            )
            return _, 0
    return _, arg1 * arg2


def _handle_product_constant_ANY(visitor, node, arg1, arg2):
    arg2[1].multiplier *= arg1[1]
    return arg2


def _handle_product_ANY_constant(visitor, node, arg1, arg2):
    arg1[1].multiplier *= arg2[1]
    return arg1


def _handle_product_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    if not visitor.expand_nonlinear_products:
        ans.nonlinear = to_expression(visitor, arg1) * to_expression(visitor, arg2)
        return _GENERAL, ans

    # We are multiplying (A + Bx + C(x)) * (A + Bx + C(x))
    _, x1 = arg1
    _, x2 = arg2
    ans.multiplier = x1.multiplier * x2.multiplier
    x1.multiplier = x2.multiplier = 1
    # x1.const * x2.const [AA]
    ans.constant = x1.constant * x2.constant
    # x1.linear * x2.const [BA] + x1.const * x2.linear [AB]
    if x2.constant:
        c = x2.constant
        if c == 1:
            ans.linear = dict(x1.linear)
        else:
            ans.linear = {vid: c * coef for vid, coef in x1.linear.items()}
    if x1.constant:
        _merge_dict(ans.linear, x1.constant, x2.linear)
    ans.nonlinear = 0
    if x1.constant and x2.nonlinear is not None:
        # [AC]
        ans.nonlinear += x1.constant * x2.nonlinear
    if x1.nonlinear is not None:
        # [CA] + [CB] + [CC]
        ans.nonlinear += x1.nonlinear * to_expression(visitor, arg2)
    if x1.linear:
        # [BB] + [BC]
        x1.constant = 0
        x1.nonlinear = None
        x2.constant = 0
        ans.nonlinear += to_expression(visitor, arg1) * to_expression(visitor, arg2)
    return _GENERAL, ans


_exit_node_handlers[ProductExpression] = {
    (_CONSTANT, _CONSTANT): _handle_product_constant_constant,
    (_CONSTANT, _LINEAR): _handle_product_constant_ANY,
    (_CONSTANT, _GENERAL): _handle_product_constant_ANY,
    (_LINEAR, _CONSTANT): _handle_product_ANY_constant,
    (_LINEAR, _LINEAR): _handle_product_nonlinear,
    (_LINEAR, _GENERAL): _handle_product_nonlinear,
    (_GENERAL, _CONSTANT): _handle_product_ANY_constant,
    (_GENERAL, _LINEAR): _handle_product_nonlinear,
    (_GENERAL, _GENERAL): _handle_product_nonlinear,
}
_exit_node_handlers[MonomialTermExpression] = _exit_node_handlers[ProductExpression]

#
# DIVISION handlers
#


def _handle_division_constant_constant(visitor, node, arg1, arg2):
    return _CONSTANT, apply_node_operation(node, (arg1[1], arg2[1]))


def _handle_division_ANY_constant(visitor, node, arg1, arg2):
    arg1[1].multiplier /= arg2[1]
    return arg1


def _handle_division_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    ans.nonlinear = to_expression(visitor, arg1) / to_expression(visitor, arg2)
    return _GENERAL, ans


_exit_node_handlers[DivisionExpression] = {
    (_CONSTANT, _CONSTANT): _handle_division_constant_constant,
    (_CONSTANT, _LINEAR): _handle_division_nonlinear,
    (_CONSTANT, _GENERAL): _handle_division_nonlinear,
    (_LINEAR, _CONSTANT): _handle_division_ANY_constant,
    (_LINEAR, _LINEAR): _handle_division_nonlinear,
    (_LINEAR, _GENERAL): _handle_division_nonlinear,
    (_GENERAL, _CONSTANT): _handle_division_ANY_constant,
    (_GENERAL, _LINEAR): _handle_division_nonlinear,
    (_GENERAL, _GENERAL): _handle_division_nonlinear,
}

#
# EXPONENTIATION handlers
#


def _handle_pow_constant_constant(visitor, node, *args):
    arg1, arg2 = args
    ans = apply_node_operation(node, (arg1[1], arg2[1]))
    if ans.__class__ in _complex_types:
        ans = complex_number_error(ans, visitor, node)
    return _CONSTANT, ans


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
    elif exp == 0:
        return _CONSTANT, 1
    else:
        return _handle_pow_nonlinear(visitor, node, arg1, arg2)


def _handle_pow_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    ans.nonlinear = to_expression(visitor, arg1) ** to_expression(visitor, arg2)
    return _GENERAL, ans


_exit_node_handlers[PowExpression] = {
    (_CONSTANT, _CONSTANT): _handle_pow_constant_constant,
    (_CONSTANT, _LINEAR): _handle_pow_nonlinear,
    (_CONSTANT, _GENERAL): _handle_pow_nonlinear,
    (_LINEAR, _CONSTANT): _handle_pow_ANY_constant,
    (_LINEAR, _LINEAR): _handle_pow_nonlinear,
    (_LINEAR, _GENERAL): _handle_pow_nonlinear,
    (_GENERAL, _CONSTANT): _handle_pow_ANY_constant,
    (_GENERAL, _LINEAR): _handle_pow_nonlinear,
    (_GENERAL, _GENERAL): _handle_pow_nonlinear,
}

#
# ABS and UNARY handlers
#


def _handle_unary_constant(visitor, node, arg):
    ans = apply_node_operation(node, (arg[1],))
    # Unary includes sqrt() which can return complex numbers
    if ans.__class__ in _complex_types:
        ans = complex_number_error(ans, visitor, node)
    return _CONSTANT, ans


def _handle_unary_nonlinear(visitor, node, arg):
    ans = visitor.Result()
    ans.nonlinear = node.create_node_with_local_data((to_expression(visitor, arg),))
    return _GENERAL, ans


_exit_node_handlers[UnaryFunctionExpression] = {
    (_CONSTANT,): _handle_unary_constant,
    (_LINEAR,): _handle_unary_nonlinear,
    (_GENERAL,): _handle_unary_nonlinear,
}
_exit_node_handlers[AbsExpression] = _exit_node_handlers[UnaryFunctionExpression]

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


_exit_node_handlers[ScalarExpression] = {
    (_CONSTANT,): _handle_named_constant,
    (_LINEAR,): _handle_named_ANY,
    (_GENERAL,): _handle_named_ANY,
}

_named_subexpression_types = [
    ScalarExpression,
    _GeneralExpressionData,
    kernel.expression.expression,
    kernel.expression.noclone,
    # Note: objectives are special named expressions
    _GeneralObjectiveData,
    ScalarObjective,
    kernel.objective.objective,
]

#
# EXPR_IF handlers
#


def _handle_expr_if_const(visitor, node, arg1, arg2, arg3):
    _type, _test = arg1
    assert _type is _CONSTANT
    if _test:
        if _test != _test or _test.__class__ is InvalidNumber:
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


_exit_node_handlers[Expr_ifExpression] = {
    (i, j, k): _handle_expr_if_nonlinear
    for i in (_LINEAR, _GENERAL)
    for j in (_CONSTANT, _LINEAR, _GENERAL)
    for k in (_CONSTANT, _LINEAR, _GENERAL)
}
for j in (_CONSTANT, _LINEAR, _GENERAL):
    for k in (_CONSTANT, _LINEAR, _GENERAL):
        _exit_node_handlers[Expr_ifExpression][_CONSTANT, j, k] = _handle_expr_if_const

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


_exit_node_handlers[EqualityExpression] = {
    (i, j): _handle_equality_general
    for i in (_CONSTANT, _LINEAR, _GENERAL)
    for j in (_CONSTANT, _LINEAR, _GENERAL)
}
_exit_node_handlers[EqualityExpression][_CONSTANT, _CONSTANT] = _handle_equality_const


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


_exit_node_handlers[InequalityExpression] = {
    (i, j): _handle_inequality_general
    for i in (_CONSTANT, _LINEAR, _GENERAL)
    for j in (_CONSTANT, _LINEAR, _GENERAL)
}
_exit_node_handlers[InequalityExpression][
    _CONSTANT, _CONSTANT
] = _handle_inequality_const


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


_exit_node_handlers[RangedExpression] = {
    (i, j, k): _handle_ranged_general
    for i in (_CONSTANT, _LINEAR, _GENERAL)
    for j in (_CONSTANT, _LINEAR, _GENERAL)
    for k in (_CONSTANT, _LINEAR, _GENERAL)
}
_exit_node_handlers[RangedExpression][
    _CONSTANT, _CONSTANT, _CONSTANT
] = _handle_ranged_const


def _before_native(visitor, child):
    return False, (_CONSTANT, child)


def _before_invalid(visitor, child):
    return False, (
        _CONSTANT,
        InvalidNumber(child, "'{child}' is not a valid numeric type"),
    )


def _before_complex(visitor, child):
    return False, (_CONSTANT, complex_number_error(child, visitor, child))


def _before_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            return False, (_CONSTANT, visitor._eval_fixed(child))
        visitor.var_map[_id] = child
        visitor.var_order[_id] = len(visitor.var_order)
    ans = visitor.Result()
    ans.linear[_id] = 1
    return False, (_LINEAR, ans)


def _before_param(visitor, child):
    return False, (_CONSTANT, visitor._eval_fixed(child))


def _before_npv(visitor, child):
    try:
        return False, (_CONSTANT, visitor._eval_expr(child))
    except (ValueError, ArithmeticError):
        return True, None


def _before_monomial(visitor, child):
    #
    # The following are performance optimizations for common
    # situations (Monomial terms and Linear expressions)
    #
    arg1, arg2 = child._args_
    if arg1.__class__ not in native_types:
        try:
            arg1 = visitor._eval_expr(arg1)
        except (ValueError, ArithmeticError):
            return True, None

    # Trap multiplication by 0 and nan.
    if not arg1:
        if arg2.fixed:
            arg2 = visitor._eval_fixed(arg2)
            if arg2 != arg2:
                deprecation_warning(
                    f"Encountered {arg1}*{str(arg2.value)} in expression "
                    "tree.  Mapping the NaN result to 0 for compatibility "
                    "with the lp_v1 writer.  In the future, this NaN "
                    "will be preserved/emitted to comply with IEEE-754.",
                    version='6.6.0',
                )
        return False, (_CONSTANT, arg1)

    _id = id(arg2)
    if _id not in visitor.var_map:
        if arg2.fixed:
            return False, (_CONSTANT, arg1 * visitor._eval_fixed(arg2))
        visitor.var_map[_id] = arg2
        visitor.var_order[_id] = len(visitor.var_order)
    ans = visitor.Result()
    ans.linear[_id] = arg1
    return False, (_LINEAR, ans)


def _before_linear(visitor, child):
    var_map = visitor.var_map
    var_order = visitor.var_order
    next_i = len(var_order)
    ans = visitor.Result()
    const = 0
    linear = ans.linear
    for arg in child.args:
        if arg.__class__ is MonomialTermExpression:
            arg1, arg2 = arg._args_
            if arg1.__class__ not in native_types:
                try:
                    arg1 = visitor._eval_expr(arg1)
                except (ValueError, ArithmeticError):
                    return True, None

            # Trap multiplication by 0 and nan.
            if not arg1:
                if arg2.fixed:
                    arg2 = visitor._eval_fixed(arg2)
                    if arg2 != arg2:
                        deprecation_warning(
                            f"Encountered {arg1}*{str(arg2.value)} in expression "
                            "tree.  Mapping the NaN result to 0 for compatibility "
                            "with the lp_v1 writer.  In the future, this NaN "
                            "will be preserved/emitted to comply with IEEE-754.",
                            version='6.6.0',
                        )
                continue

            _id = id(arg2)
            if _id not in var_map:
                if arg2.fixed:
                    const += arg1 * visitor._eval_fixed(arg2)
                    continue
                var_map[_id] = arg2
                var_order[_id] = next_i
                next_i += 1
                linear[_id] = arg1
            elif _id in linear:
                linear[_id] += arg1
            else:
                linear[_id] = arg1
        elif arg.__class__ in native_numeric_types:
            const += arg
        else:
            try:
                const += visitor._eval_expr(arg)
            except (ValueError, ArithmeticError):
                return True, None
    if linear:
        ans.constant = const
        return False, (_LINEAR, ans)
    else:
        return False, (_CONSTANT, const)


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


def _before_external(visitor, child):
    ans = visitor.Result()
    if all(is_fixed(arg) for arg in child.args):
        try:
            ans.constant = visitor._eval_expr(child)
            return False, (_CONSTANT, ans)
        except:
            pass
    ans.nonlinear = child
    return False, (_GENERAL, ans)


def _before_general_expression(visitor, child):
    return True, None


def _register_new_before_child_dispatcher(visitor, child):
    dispatcher = _before_child_dispatcher
    child_type = child.__class__
    if child_type in native_numeric_types:
        if issubclass(child_type, complex):
            _complex_types.add(child_type)
            dispatcher[child_type] = _before_complex
        else:
            dispatcher[child_type] = _before_native
    elif child_type in native_types:
        dispatcher[child_type] = _before_invalid
    elif not child.is_expression_type():
        if child.is_potentially_variable():
            dispatcher[child_type] = _before_var
        else:
            dispatcher[child_type] = _before_param
    elif not child.is_potentially_variable():
        dispatcher[child_type] = _before_npv
        # If we descend into the named expression (because of an
        # evaluation error), then on the way back out, we will use
        # the potentially variable handler to process the result.
        pv_base_type = child.potentially_variable_base_class()
        if pv_base_type not in dispatcher:
            try:
                child.__class__ = pv_base_type
                _register_new_before_child_dispatcher(visitor, child)
            finally:
                child.__class__ = child_type
        if pv_base_type in visitor.exit_node_handlers:
            visitor.exit_node_handlers[child_type] = visitor.exit_node_handlers[
                pv_base_type
            ]
            for args, fcn in visitor.exit_node_handlers[child_type].items():
                visitor.exit_node_dispatcher[(child_type, *args)] = fcn
    elif id(child) in visitor.subexpression_cache or issubclass(
        child_type, _GeneralExpressionData
    ):
        dispatcher[child_type] = _before_named_expression
        visitor.exit_node_handlers[child_type] = visitor.exit_node_handlers[
            ScalarExpression
        ]
        for args, fcn in visitor.exit_node_handlers[child_type].items():
            visitor.exit_node_dispatcher[(child_type, *args)] = fcn
    else:
        dispatcher[child_type] = _before_general_expression
    return dispatcher[child_type](visitor, child)


_before_child_dispatcher = collections.defaultdict(
    lambda: _register_new_before_child_dispatcher
)

# For efficiency reasons, we will maintain a separate list of all
# complex number types
_complex_types = set((complex,))

# We do not support writing complex numbers out
_before_child_dispatcher[complex] = _before_complex
# Special handling for external functions: will be handled
# as terminal nodes from the point of view of the visitor
_before_child_dispatcher[ExternalFunctionExpression] = _before_external
# Special linear / summation expressions
_before_child_dispatcher[MonomialTermExpression] = _before_monomial
_before_child_dispatcher[LinearExpression] = _before_linear
_before_child_dispatcher[SumExpression] = _before_general_expression


#
# Initialize the _exit_node_dispatcher
#
def _initialize_exit_node_dispatcher(exit_handlers):
    # expand the knowns set of named expressiosn
    for expr in _named_subexpression_types:
        exit_handlers[expr] = exit_handlers[ScalarExpression]

    exit_dispatcher = {}
    for cls, handlers in exit_handlers.items():
        for args, fcn in handlers.items():
            exit_dispatcher[(cls, *args)] = fcn
    return exit_dispatcher


class LinearRepnVisitor(StreamBasedExpressionVisitor):
    Result = LinearRepn
    exit_node_handlers = _exit_node_handlers
    exit_node_dispatcher = _initialize_exit_node_dispatcher(_exit_node_handlers)
    expand_nonlinear_products = False
    max_exponential_expansion = 1

    def __init__(self, subexpression_cache, var_map, var_order):
        super().__init__()
        self.subexpression_cache = subexpression_cache
        self.var_map = var_map
        self.var_order = var_order
        self._eval_expr_visitor = _EvaluationVisitor(True)

    def _eval_fixed(self, obj):
        ans = obj.value
        if ans.__class__ not in native_numeric_types:
            # None can be returned from uninitialized Var/Param objects
            if ans is None:
                return InvalidNumber(
                    None, f"'{obj}' contains a nonnumeric value '{ans}'"
                )
            if ans.__class__ is InvalidNumber:
                return ans
            else:
                # It is possible to get other non-numeric types.  Most
                # common are bool and 1-element numpy.array().  We will
                # attempt to convert the value to a float before
                # proceeding.
                #
                # TODO: we should check bool and warn/error (while bool is
                # convertible to float in Python, they have very
                # different semantic meanings in Pyomo).
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(
                        ans, f"'{obj}' contains a nonnumeric value '{ans}'"
                    )
        if ans != ans:
            return InvalidNumber(nan, f"'{obj}' contains a nonnumeric value '{ans}'")
        if ans.__class__ in _complex_types:
            return complex_number_error(ans, self, obj)
        return ans

    def _eval_expr(self, expr):
        ans = self._eval_expr_visitor.dfs_postorder_stack(expr)
        if ans.__class__ not in native_numeric_types:
            # None can be returned from uninitialized Expression objects
            if ans is None:
                return InvalidNumber(
                    ans, f"'{expr}' evaluated to nonnumeric value '{ans}'"
                )
            if ans.__class__ is InvalidNumber:
                return ans
            else:
                # It is possible to get other non-numeric types.  Most
                # common are bool and 1-element numpy.array().  We will
                # attempt to convert the value to a float before
                # proceeding.
                #
                # TODO: we should check bool and warn/error (while bool is
                # convertible to float in Python, they have very
                # different semantic meanings in Pyomo).
                try:
                    ans = float(ans)
                except:
                    return InvalidNumber(
                        ans, f"'{expr}' evaluated to nonnumeric value '{ans}'"
                    )
        if ans != ans:
            return InvalidNumber(ans, f"'{expr}' evaluated to nonnumeric value '{ans}'")
        if ans.__class__ in _complex_types:
            return complex_number_error(ans, self, expr)
        return ans

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        return _before_child_dispatcher[child.__class__](self, child)

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
        if ans.__class__ is self.Result:
            mult = ans.multiplier
            if mult == 1:
                # mult is identity: only thing to do is filter out zero coefficients
                zeros = list(filterfalse(itemgetter(1), ans.linear.items()))
                for vid, coef in zeros:
                    del ans.linear[vid]
            elif not mult:
                # the mulltiplier has cleared out the entire expression.
                # Warn if this is suppressing a NaN (unusual, and
                # non-standard, but we will wait to remove this behavior
                # for the time being)
                if ans.constant != ans.constant or any(
                    c != c for c in ans.linear.values()
                ):
                    deprecation_warning(
                        f"Encountered {str(mult)}*nan in expression tree.  "
                        "Mapping the NaN result to 0 for compatibility "
                        "with the lp_v1 writer.  In the future, this NaN "
                        "will be preserved/emitted to comply with IEEE-754.",
                        version='6.6.0',
                    )
                return self.Result()
            else:
                # mult not in {0, 1}: factor it into the constant,
                # linear coefficients, and nonlinear term
                linear = ans.linear
                zeros = []
                for vid, coef in linear.items():
                    if coef:
                        linear[vid] = coef * mult
                    else:
                        zeros.append(vid)
                for vid in zeros:
                    del linear[vid]
                if ans.nonlinear is not None:
                    ans.nonlinear *= mult
                if ans.constant:
                    ans.constant *= mult
                ans.multiplier = 1
            return ans
        ans = self.Result()
        assert result[0] is _CONSTANT
        ans.constant = result[1]
        return ans
