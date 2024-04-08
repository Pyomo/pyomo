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
            print("NORMAL: %s" % child)
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
            print("DATA: %s" % child)
            # We aren't treating this Var as a Var for the purposes of this walker
            return False, (ExprType.CONSTANT, child)


_before_child_dispatcher = MultiLevelLinearBeforeChildDispatcher()


class MultilevelLinearRepnVisitor(LinearRepnVisitor):
    def __init__(self, subexpression_cache, var_map, var_order, sorter, wrt):
        super().__init__(subexpression_cache, var_map, var_order, sorter)
        self.wrt = ComponentSet(_flattened(wrt))

    def beforeChild(self, node, child, child_idx):
        print("before child %s" % child)
        print(child.__class__)
        return _before_child_dispatcher[child.__class__](self, child)
