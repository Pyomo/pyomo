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

from pyomo.contrib.piecewise.piecewise_linear_expression import (
    PiecewiseLinearExpression,
)
from pyomo.core import Expression
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor


class PiecewiseLinearToMIP(StreamBasedExpressionVisitor):
    """
    Expression walker to replace PiecewiseLinearExpressions when creating
    equivalent MIP formulations.

    Args:
        transform_pw_linear_expression (function): a callback that accepts
            a PiecewiseLinearExpression, its parent PiecewiseLinearFunction,
            and a transformation Block. It is expected to convert the
            PiecewiseLinearExpression to MIP form, and return the Var (or
            other expression) that will replace it in the expression.
        transBlock (Block): transformation Block to pass to the above
            callback
    """

    def __init__(self, transform_pw_linear_expression, transBlock):
        self.transform_pw_linear_expression = transform_pw_linear_expression
        self.transBlock = transBlock
        self._process_node = self._process_node_bx

    def initializeWalker(self, expr):
        expr, src, src_idx = expr
        # always walk
        return True, expr

    def beforeChild(self, node, child, child_idx):
        return True, None

    def exitNode(self, node, data):
        if node.__class__ is PiecewiseLinearExpression:
            parent = node.pw_linear_function
            substitute_var = self.transform_pw_linear_expression(
                node, parent, self.transBlock
            )
            parent._expressions[parent._expression_ids[node]] = substitute_var
        return node

    finalizeResult = None
