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

from pyomo.common.collections import ComponentSet
from pyomo.common.numeric_types import native_numeric_types
from pyomo.core import Var
from pyomo.core.expr.logical_expr import _flattened
from pyomo.core.expr.numeric_expr import (
    LinearExpression,
    MonomialTermExpression,
    SumExpression,
)
from pyomo.repn.linear import LinearBeforeChildDispatcher, LinearRepnVisitor
from pyomo.repn.util import ExprType


class MultiLevelLinearBeforeChildDispatcher(LinearBeforeChildDispatcher):
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
        if child in visitor.wrt:
            # This is a normal situation
            _id = id(child)
            if _id not in visitor.var_map:
                if child.fixed:
                    return False, (
                        ExprType.CONSTANT,
                        visitor.check_constant(child.value, child),
                    )
                MultiLevelLinearBeforeChildDispatcher._record_var(visitor, child)
            ans = visitor.Result()
            ans.linear[_id] = 1
            return False, (ExprType.LINEAR, ans)
        else:
            # We aren't treating this Var as a Var for the purposes of this walker
            return False, (ExprType.CONSTANT, child)


_before_child_dispatcher = MultiLevelLinearBeforeChildDispatcher()


# LinearSubsystemRepnVisitor
class MultilevelLinearRepnVisitor(LinearRepnVisitor):
    def __init__(self, subexpression_cache, var_map, var_order, sorter, wrt):
        super().__init__(subexpression_cache, var_map, var_order, sorter)
        self.wrt = ComponentSet(_flattened(wrt))

    def beforeChild(self, node, child, child_idx):
        return _before_child_dispatcher[child.__class__](self, child)

    def finalizeResult(self, result):
        ans = result[1]
        if ans.__class__ is self.Result:
            mult = ans.multiplier
            if mult.__class__ not in native_numeric_types:
                # mult is an expression--we should push it back into the other terms
                self._factor_multiplier_into_linear_terms(ans, mult)
                return ans
            if mult == 1:
                zeros = [(vid, coef) for vid, coef in ans.linear.items() if
                         coef.__class__ in native_numeric_types and not coef]
                for vid, coef in zeros:
                    del ans.linear[vid]
            elif not mult:
                # the mulltiplier has cleared out the entire expression.
                # Warn if this is suppressing a NaN (unusual, and
                # non-standard, but we will wait to remove this behavior
                # for the time being)
                # ESJ TODO: This won't work either actually...
                # I'm not sure how to do it.
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
                self._factor_multiplier_into_linear_terms(ans, mult)
            return ans

        ans = self.Result()
        assert result[0] is ExprType.CONSTANT
        ans.constant = result[1]
        return ans
