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

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import (
    ExternalFunctionExpression,
    NegationExpression,
    PowExpression,
    MaxExpression,
    MinExpression,
    ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    SumExpression,
    Expr_ifExpression,
    UnaryFunctionExpression,
    AbsExpression,
)
from pyomo.core.expr.relational_expr import (
    RangedExpression,
    InequalityExpression,
    EqualityExpression,
)
from pyomo.core.base.var import VarData, ScalarVar
from pyomo.core.base.param import ParamData, ScalarParam
from pyomo.core.base.expression import ExpressionData, ScalarExpression
from pyomo.repn.util import ExitNodeDispatcher
from pyomo.common.collections import ComponentSet


def handle_var(node, collector):
    collector.variables.add(node)
    return None


def handle_param(node, collector):
    collector.params.add(node)
    return None


def handle_named_expression(node, collector):
    collector.named_expressions.add(node)
    return None


def handle_external_function(node, collector):
    collector.external_functions.add(node)
    return None


def handle_skip(node, collector):
    return None


collector_handlers = ExitNodeDispatcher()
collector_handlers[VarData] = handle_var
collector_handlers[ParamData] = handle_param
collector_handlers[ExpressionData] = handle_named_expression
collector_handlers[ScalarExpression] = handle_named_expression
collector_handlers[ExternalFunctionExpression] = handle_external_function
collector_handlers[NegationExpression] = handle_skip
collector_handlers[PowExpression] = handle_skip
collector_handlers[MaxExpression] = handle_skip
collector_handlers[MinExpression] = handle_skip
collector_handlers[ProductExpression] = handle_skip
collector_handlers[MonomialTermExpression] = handle_skip
collector_handlers[DivisionExpression] = handle_skip
collector_handlers[SumExpression] = handle_skip
collector_handlers[Expr_ifExpression] = handle_skip
collector_handlers[UnaryFunctionExpression] = handle_skip
collector_handlers[AbsExpression] = handle_skip
collector_handlers[RangedExpression] = handle_skip
collector_handlers[InequalityExpression] = handle_skip
collector_handlers[EqualityExpression] = handle_skip
collector_handlers[int] = handle_skip
collector_handlers[float] = handle_skip


class _ComponentFromExprCollector(StreamBasedExpressionVisitor):
    def __init__(self, **kwds):
        self.named_expressions = ComponentSet()
        self.variables = ComponentSet()
        self.params = ComponentSet()
        self.external_functions = ComponentSet()
        super().__init__(**kwds)

    def exitNode(self, node, data):
        return collector_handlers[node.__class__](node, self)

    def beforeChild(self, node, child, child_idx):
        if child in self.named_expressions:
            return False, None
        return True, None


_visitor = _ComponentFromExprCollector()


def collect_components_from_expr(expr):
    _visitor.__init__()
    _visitor.walk_expression(expr)
    return (
        _visitor.named_expressions,
        _visitor.variables,
        _visitor.params,
        _visitor.external_functions,
    )
