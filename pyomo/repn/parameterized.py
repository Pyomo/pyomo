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

from pyomo.common.collections import ComponentSet
from pyomo.common.numeric_types import native_numeric_types
from pyomo.core import Var
from pyomo.core.expr.logical_expr import _flattened
from pyomo.core.expr.numeric_expr import (
    LinearExpression,
    MonomialTermExpression,
    ProductExpression,
    SumExpression,
)
from pyomo.repn.util import (
    ExitNodeDispatcher,
    ExprType,
    initialize_exit_node_dispatcher,
)
import pyomo.repn.linear as linear
import pyomo.repn.quadratic as quadratic


_FIXED = ExprType.FIXED
_CONSTANT = ExprType.CONSTANT
_LINEAR = ExprType.LINEAR
_GENERAL = ExprType.GENERAL


class ParameterizedRepnMixin(object):
    @staticmethod
    def constant_flag(val):
        if val.__class__ in native_numeric_types:
            return val
        return 2  # something not 0 or 1

    @staticmethod
    def multiplier_flag(val):
        if val.__class__ in native_numeric_types:
            if not val:
                return 2
            return val
        return 2  # something not 0 or 1

    def walker_exitNode(self):
        ans = super().walker_exitNode()
        if ans[0] is _CONSTANT and (
            self.constant.__class__ not in native_numeric_types
            or self.multiplier.__class__ not in native_numeric_types
        ):
            return _FIXED, ans[1]
        return ans


class ParameterizedBeforeChildDispatcher(linear.LinearBeforeChildDispatcher):
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
            if child in visitor.wrt:
                # pseudo-constant, and we need to leave it in even if it is fixed!
                # We aren't treating this Var as a Var for the purposes of this walker
                return False, (_FIXED, child)
            if child.fixed:
                return False, (_CONSTANT, visitor.check_constant(child.value, child))
            # This is a normal situation
            visitor.var_recorder.add(child)
        ans = visitor.Result()
        ans.linear[_id] = 1
        return False, (ExprType.LINEAR, ans)


def _handle_product_constant_constant(visitor, node, arg1, arg2):
    # [ESJ 5/22/24]: Overriding this handler to exclude the deprecation path for
    # 0 * nan. It doesn't need overridden when that deprecation path goes away.
    return _CONSTANT, arg1[1] * arg2[1]


def update_exit_node_handlers(exit_node_handlers=None):
    exit_node_handlers[ProductExpression][
        (_CONSTANT, _CONSTANT)
    ] = _handle_product_constant_constant
    return exit_node_handlers


class ParameterizedRepnVisitorMixin:
    def __init__(
        self,
        subexpression_cache,
        var_map=None,
        var_order=None,
        sorter=None,
        wrt=None,
        var_recorder=None,
    ):
        super().__init__(
            subexpression_cache=subexpression_cache,
            var_map=var_map,
            var_order=var_order,
            sorter=sorter,
            var_recorder=var_recorder,
        )
        if wrt is None:
            raise ValueError(f"{self.__class__.__name__}: wrt not specified")
        self.wrt = ComponentSet(_flattened(wrt))

    def finalizeResult(self, result):
        ans = result[1]
        if ans.__class__ is self.Result and not ans.constant_flag(ans.multiplier):
            self._factor_multiplier_into_ans(ans, 0)
            return ans
        return super().finalizeResult(result)


class ParameterizedLinearRepn(ParameterizedRepnMixin, linear.LinearRepn):
    pass


class ParameterizedLinearRepnVisitor(
    ParameterizedRepnVisitorMixin, linear.LinearRepnVisitor
):
    Result = ParameterizedLinearRepn
    before_child_dispatcher = ParameterizedBeforeChildDispatcher()
    exit_node_dispatcher = ExitNodeDispatcher(
        linear.initialize_exit_node_dispatcher(
            update_exit_node_handlers(linear.define_exit_node_handlers())
        )
    )


class ParameterizedQuadraticRepn(ParameterizedRepnMixin, quadratic.QuadraticRepn):
    pass


class ParameterizedQuadraticRepnVisitor(
    ParameterizedRepnVisitorMixin, quadratic.QuadraticRepnVisitor
):
    Result = ParameterizedQuadraticRepn
    before_child_dispatcher = ParameterizedBeforeChildDispatcher()
    exit_node_dispatcher = ExitNodeDispatcher(
        initialize_exit_node_dispatcher(
            update_exit_node_handlers(quadratic.define_exit_node_handlers())
        )
    )
