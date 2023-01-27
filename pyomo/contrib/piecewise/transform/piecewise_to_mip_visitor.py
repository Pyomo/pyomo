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

from pyomo.contrib.piecewise.piecewise_linear_expression import (
    PiecewiseLinearExpression)
from pyomo.core import Expression
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor

class PiecewiseLinearToMIP(StreamBasedExpressionVisitor):
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
            parent = node.parent_pw_linear_function
            substitute_var = self.transform_pw_linear_expression(
                node, parent, self.transBlock)
            parent._expressions[node._index] = substitute_var
        return node

    finalizeResult = None
