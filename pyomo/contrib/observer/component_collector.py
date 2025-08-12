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
    NPV_ExternalFunctionExpression,
)
from pyomo.core.base.var import VarData, ScalarVar
from pyomo.core.base.param import ParamData, ScalarParam
from pyomo.core.base.expression import ExpressionData, ScalarExpression


def handle_var(node, collector):
    collector.variables[id(node)] = node
    return None


def handle_param(node, collector):
    collector.params[id(node)] = node
    return None


def handle_named_expression(node, collector):
    collector.named_expressions[id(node)] = node
    return None


def handle_external_function(node, collector):
    collector.external_functions[id(node)] = node
    return None


collector_handlers = {
    VarData: handle_var,
    ScalarVar: handle_var,
    ParamData: handle_param,
    ScalarParam: handle_param,
    ExpressionData: handle_named_expression,
    ScalarExpression: handle_named_expression,
    ExternalFunctionExpression: handle_external_function,
    NPV_ExternalFunctionExpression: handle_external_function,
}


class _ComponentFromExprCollector(StreamBasedExpressionVisitor):
    def __init__(self, **kwds):
        self.named_expressions = {}
        self.variables = {}
        self.params = {}
        self.external_functions = {}
        super().__init__(**kwds)

    def exitNode(self, node, data):
        nt = type(node)
        if nt in collector_handlers:
            return collector_handlers[nt](node, self)
        else:
            return None


_visitor = _ComponentFromExprCollector()


def collect_components_from_expr(expr):
    _visitor.__init__()
    _visitor.walk_expression(expr)
    return (
        list(_visitor.named_expressions.values()),
        list(_visitor.variables.values()),
        list(_visitor.params.values()),
        list(_visitor.external_functions.values()),
    )
