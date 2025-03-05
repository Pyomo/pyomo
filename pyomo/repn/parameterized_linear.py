#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy

from pyomo.common.collections import ComponentSet
from pyomo.common.numeric_types import native_numeric_types
from pyomo.core import Var
from pyomo.core.expr.logical_expr import _flattened
from pyomo.core.expr.numeric_expr import (
    AbsExpression,
    DivisionExpression,
    LinearExpression,
    MonomialTermExpression,
    NegationExpression,
    mutable_expression,
    PowExpression,
    ProductExpression,
    SumExpression,
    UnaryFunctionExpression,
)
from pyomo.repn.linear import (
    ExitNodeDispatcher,
    _initialize_exit_node_dispatcher,
    LinearBeforeChildDispatcher,
    LinearRepn,
    LinearRepnVisitor,
)
from pyomo.repn.util import ExprType
import pyomo.repn.linear as linear


_FIXED = ExprType.FIXED
_CONSTANT = ExprType.CONSTANT
_LINEAR = ExprType.LINEAR
_GENERAL = ExprType.GENERAL


def _merge_dict(dest_dict, mult, src_dict):
    if mult.__class__ not in native_numeric_types or mult != 1:
        for vid, coef in src_dict.items():
            if vid in dest_dict:
                dest_dict[vid] += mult * coef
            else:
                dest_dict[vid] = mult * coef
    else:
        for vid, coef in src_dict.items():
            if vid in dest_dict:
                dest_dict[vid] += coef
            else:
                dest_dict[vid] = coef


def to_expression(visitor, arg):
    if arg[0] in (_CONSTANT, _FIXED):
        return arg[1]
    else:
        return arg[1].to_expression(visitor)


class ParameterizedLinearRepn(LinearRepn):
    def __str__(self):
        return (
            f"ParameterizedLinearRepn(mult={self.multiplier}, const={self.constant}, "
            f"linear={self.linear}, nonlinear={self.nonlinear})"
        )

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return _GENERAL, self
        elif self.linear:
            return _LINEAR, self
        elif self.constant.__class__ in native_numeric_types:
            return _CONSTANT, self.multiplier * self.constant
        else:
            return _FIXED, self.multiplier * self.constant

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
            with mutable_expression() as e:
                for vid, coef in self.linear.items():
                    if coef.__class__ not in native_numeric_types or coef:
                        e += coef * var_map[vid]
            if e.nargs() > 1:
                ans += e
            elif e.nargs() == 1:
                ans += e.arg(0)
        if self.constant.__class__ not in native_numeric_types or self.constant:
            ans += self.constant
        if (
            self.multiplier.__class__ not in native_numeric_types
            or self.multiplier != 1
        ):
            ans *= self.multiplier
        return ans

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a ParameterizedLinearRepn() as a `data` object in
        the expression walker (thereby allowing us to use the default
        implementation of acceptChildResult [which calls
        `data.append()`] and avoid the function call for a custom
        callback).

        """
        _type, other = other
        if _type is _CONSTANT or _type is _FIXED:
            self.constant += other
            return

        mult = other.multiplier
        try:
            _mult = bool(mult)
            if not _mult:
                return
            if mult == 1:
                _mult = False
        except:
            _mult = True

        const = other.constant
        try:
            _const = bool(const)
        except:
            _const = True

        if _mult:
            if _const:
                self.constant += mult * const
            if other.linear:
                _merge_dict(self.linear, mult, other.linear)
            if other.nonlinear is not None:
                nl = mult * other.nonlinear
                if self.nonlinear is None:
                    self.nonlinear = nl
                else:
                    self.nonlinear += nl
        else:
            if _const:
                self.constant += const
            if other.linear:
                _merge_dict(self.linear, 1, other.linear)
            if other.nonlinear is not None:
                nl = other.nonlinear
                if self.nonlinear is None:
                    self.nonlinear = nl
                else:
                    self.nonlinear += nl


class ParameterizedLinearBeforeChildDispatcher(LinearBeforeChildDispatcher):
    def __init__(self):
        super().__init__()
        self[Var] = self._before_var
        self[MonomialTermExpression] = self._before_monomial
        self[LinearExpression] = self._before_linear
        self[SumExpression] = self._before_general_expression

    @staticmethod
    def _before_linear(visitor, child):
        return True, None

    @staticmethod
    def _before_monomial(visitor, child):
        return True, None

    @staticmethod
    def _before_general_expression(visitor, child):
        return True, None

    @staticmethod
    def _before_var(visitor, child):
        _id = id(child)
        if _id not in visitor.var_map:
            if child.fixed:
                return False, (_CONSTANT, visitor.check_constant(child.value, child))
            if child in visitor.wrt:
                # pseudo-constant
                # We aren't treating this Var as a Var for the purposes of this walker
                return False, (_FIXED, child)
            # This is a normal situation
            ParameterizedLinearBeforeChildDispatcher._record_var(visitor, child)
        ans = visitor.Result()
        ans.linear[_id] = 1
        return False, (ExprType.LINEAR, ans)


_before_child_dispatcher = ParameterizedLinearBeforeChildDispatcher()

#
# NEGATION handlers
#


def _handle_negation_pseudo_constant(visitor, node, arg):
    return (_FIXED, -1 * arg[1])


#
# PRODUCT handlers
#


def _handle_product_constant_constant(visitor, node, arg1, arg2):
    # [ESJ 5/22/24]: Overriding this handler to exclude the deprecation path for
    # 0 * nan. It doesn't need overridden when that deprecation path goes away.
    return _CONSTANT, arg1[1] * arg2[1]


def _handle_product_pseudo_constant_constant(visitor, node, arg1, arg2):
    return _FIXED, arg1[1] * arg2[1]


#
# DIVISION handlers
#


def _handle_division_pseudo_constant_constant(visitor, node, arg1, arg2):
    return _FIXED, arg1[1] / arg2[1]


def _handle_division_ANY_pseudo_constant(visitor, node, arg1, arg2):
    arg1[1].multiplier = arg1[1].multiplier / arg2[1]
    return arg1


#
# EXPONENTIATION handlers
#


def _handle_pow_pseudo_constant_constant(visitor, node, arg1, arg2):
    return _FIXED, to_expression(visitor, arg1) ** to_expression(visitor, arg2)


def _handle_pow_nonlinear(visitor, node, arg1, arg2):
    # ESJ: We override this because we need our own to_expression implementation
    # if pseudo constants are involved.
    ans = visitor.Result()
    ans.nonlinear = to_expression(visitor, arg1) ** to_expression(visitor, arg2)
    return _GENERAL, ans


#
# ABS and UNARY handlers
#


def _handle_unary_pseudo_constant(visitor, node, arg):
    # We override this because we can't blindly use apply_node_operation in this case
    return _FIXED, node.create_node_with_local_data((to_expression(visitor, arg),))


def define_exit_node_handlers(exit_node_handlers=None):
    if exit_node_handlers is None:
        exit_node_handlers = {}
    linear.define_exit_node_handlers(exit_node_handlers)

    exit_node_handlers[NegationExpression].update(
        {(_FIXED,): _handle_negation_pseudo_constant}
    )

    exit_node_handlers[ProductExpression].update(
        {
            (_CONSTANT, _CONSTANT): _handle_product_constant_constant,
            (_FIXED, _FIXED): _handle_product_pseudo_constant_constant,
            (_FIXED, _CONSTANT): _handle_product_pseudo_constant_constant,
            (_CONSTANT, _FIXED): _handle_product_pseudo_constant_constant,
            (_FIXED, _LINEAR): linear._handle_product_constant_ANY,
            (_LINEAR, _FIXED): linear._handle_product_ANY_constant,
            (_FIXED, _GENERAL): linear._handle_product_constant_ANY,
            (_GENERAL, _FIXED): linear._handle_product_ANY_constant,
        }
    )

    exit_node_handlers[MonomialTermExpression].update(
        exit_node_handlers[ProductExpression]
    )

    exit_node_handlers[DivisionExpression].update(
        {
            (_FIXED, _FIXED): _handle_division_pseudo_constant_constant,
            (_FIXED, _CONSTANT): _handle_division_pseudo_constant_constant,
            (_CONSTANT, _FIXED): _handle_division_pseudo_constant_constant,
            (_LINEAR, _FIXED): _handle_division_ANY_pseudo_constant,
            (_GENERAL, _FIXED): _handle_division_ANY_pseudo_constant,
        }
    )

    exit_node_handlers[PowExpression].update(
        {
            (_FIXED, _FIXED): _handle_pow_pseudo_constant_constant,
            (_FIXED, _CONSTANT): _handle_pow_pseudo_constant_constant,
            (_CONSTANT, _FIXED): _handle_pow_pseudo_constant_constant,
            (_LINEAR, _FIXED): _handle_pow_nonlinear,
            (_FIXED, _LINEAR): _handle_pow_nonlinear,
            (_GENERAL, _FIXED): _handle_pow_nonlinear,
            (_FIXED, _GENERAL): _handle_pow_nonlinear,
        }
    )
    exit_node_handlers[UnaryFunctionExpression].update(
        {(_FIXED,): _handle_unary_pseudo_constant}
    )
    exit_node_handlers[AbsExpression] = exit_node_handlers[UnaryFunctionExpression]

    return exit_node_handlers


class ParameterizedLinearRepnVisitor(LinearRepnVisitor):
    Result = ParameterizedLinearRepn
    exit_node_dispatcher = ExitNodeDispatcher(
        _initialize_exit_node_dispatcher(define_exit_node_handlers())
    )

    def __init__(self, subexpression_cache, var_map, var_order, sorter, wrt):
        super().__init__(subexpression_cache, var_map, var_order, sorter)
        self.wrt = ComponentSet(_flattened(wrt))

    def beforeChild(self, node, child, child_idx):
        return _before_child_dispatcher[child.__class__](self, child)

    def _factor_multiplier_into_linear_terms(self, ans, mult):
        linear = ans.linear
        zeros = []
        for vid, coef in linear.items():
            if coef.__class__ not in native_numeric_types or coef:
                linear[vid] = mult * coef
            else:
                zeros.append(vid)
        for vid in zeros:
            del linear[vid]
        if ans.nonlinear is not None:
            ans.nonlinear *= mult
        if ans.constant.__class__ not in native_numeric_types or ans.constant:
            ans.constant *= mult
        ans.multiplier = 1

    def finalizeResult(self, result):
        ans = result[1]
        if ans.__class__ is self.Result:
            mult = ans.multiplier
            if mult.__class__ not in native_numeric_types:
                # mult is an expression--we should push it back into the other terms
                self._factor_multiplier_into_linear_terms(ans, mult)
                return ans
            if mult == 1:
                zeros = [
                    (vid, coef)
                    for vid, coef in ans.linear.items()
                    if coef.__class__ in native_numeric_types and not coef
                ]
                for vid, coef in zeros:
                    del ans.linear[vid]
            elif not mult:
                # the multiplier has cleared out the entire expression.  Check
                # if this is suppressing a NaN because we can't clear everything
                # out if it is
                if ans.constant != ans.constant or any(
                    c != c for c in ans.linear.values()
                ):
                    # There's a nan in here, so we distribute the 0
                    self._factor_multiplier_into_linear_terms(ans, mult)
                    return ans
                return self.Result()
            else:
                # mult not in {0, 1}: factor it into the constant,
                # linear coefficients, and nonlinear term
                self._factor_multiplier_into_linear_terms(ans, mult)
            return ans

        ans = self.Result()
        assert result[0] in (_CONSTANT, _FIXED)
        ans.constant = result[1]
        return ans
