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

from pyomo.common.collections import ComponentMap
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
import pyomo.core.expr.current as EXPR
import pyomo.core.expr.logical_expr as LE
from pyomo.core.base import Binary, VarList, ConstraintList
import pyomo.core.base.boolean_var as BV
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.var import ScalarVar, _GeneralVarData
from pyomo.gdp.disjunct import AutoLinkedBooleanVar

def _dispatch_boolean_var(visitor, node):
    if node not in visitor.boolean_to_binary_map:
        binary = node.get_associated_binary()
        if binary is not None:
            visitor.boolean_to_binary_map[node] = binary
        else:
            z = visitor.z_vars.add()
            visitor.boolean_to_binary_map[node] = z
            node.associate_binary_var(z)
    return False, visitor.boolean_to_binary_map[node]

def _dispatch_var(visitor, node):
    return False, node

def _dispatch_expression(visitor, node):
    return False, node.expr

def _dispatch_not(visitor, node, a):
    # z == !a
    if a not in visitor.expansions:
        z = visitor.z_vars.add()
        visitor.constraints.add(z == 1 - a)
        visitor.expansions[a] = z
    return visitor.expansions[a]

def _dispatch_implication(visitor, node, a, b):
    # z == !a v b
    return _dispatch_or(visitor, node, 1 - a, b)

def _dispatch_equivalence(visitor, node, a, b):
    # z == (!a v b) ^ (a v !b)
    return _dispatch_and(
        visitor, node,
        _dispatch_or(visitor, node, 1 - a, b),
        _dispatch_or(visitor, node, a, 1 - b),
    )

def _dispatch_and(visitor, node, *args):
    # z == a ^ b ^ ...
    z = visitor.z_vars.add()
    for arg in args:
        visitor.constraints.add(arg >= z)
    return z

def _dispatch_or(visitor, node, *args):
    # z == a v b v ...
    # (!z v a v b v ...) ^ (z v !a) ^ (z v !b) ^ ...
    z = visitor.z_vars.add()
    visitor.constraints.add((1 - z) + sum(args) >= 1)
    for arg in args:
        visitor.constraints.add(z + (1 - arg) >= 1)
    return z

def _dispatch_xor(visitor, node, a, b):
    # z == a XOR b
    # This is a special case of exactly
    return _dispatch_exactly(visitor, node, 1, a, b)

def _dispatch_exactly(visitor, node, *args):
    # z = sum(args[1:] == args[0]
    # This is currently implemented as a big-m transformation of:
    # [sum(args[1:] = n] v [[sum(args[1:]) < n] v [sum(args[1:]) > n]]
    z = visitor.z_vars.add()
    M = len(args) - 1
    n = args[0]
    sum_expr = sum(args[1:])
    visitor.constraints.add(sum_expr <= n + (M - n)*(1 - z))
    visitor.constraints.add(sum_expr >= n - n*(1 - z))
    a = visitor.z_vars.add()
    b = visitor.z_vars.add()
    visitor.constraints.add(1 - z == a + b)
    visitor.constraints.add(sum_expr >= n + 1 - (n + 1)*(1 - a))
    visitor.constraints.add(sum_expr <= n - 1 + (M - n + 1)*(1 - b))
    return z

def _dispatch_atleast(visitor, node, *args):
    # z = sum[args[1:] >= n
    # This is implemented as a big-m transformation of:
    # [sum(args[1:] >= n] v [sum(args[1:] < n]
    z = visitor.z_vars.add()
    n = args[0]
    sum_expr = sum(args[1:])
    visitor.constraints.add(sum_expr >= n - n*(1 - z))
    visitor.constraints.add(sum_expr <= n - 1 + (len(args) - n)*z)
    return z

def _dispatch_atmost(visitor, node, *args):
    # z = sum[args[1:] <= n
    # This is implemented as a big-m transformation of:
    # [sum(args[1:] <= n] v [sum(args[1:] > n]
    z = visitor.z_vars.add()
    n = args[0]
    sum_expr = sum(args[1:])
    visitor.constraints.add(sum_expr <= n + (len(args) - 1 - n)*(1 - z))
    visitor.constraints.add(sum_expr >= n + 1 - (n + 1)*z)
    return z

#_operator_dispatcher = collections.defaultdict(_register_dispatcher_type)
_operator_dispatcher = {}
_operator_dispatcher[LE.ImplicationExpression] = _dispatch_implication
_operator_dispatcher[LE.EquivalenceExpression] = _dispatch_equivalence
_operator_dispatcher[LE.NotExpression] = _dispatch_not
_operator_dispatcher[LE.AndExpression] = _dispatch_and
_operator_dispatcher[LE.OrExpression] = _dispatch_or
_operator_dispatcher[LE.XorExpression] = _dispatch_xor
_operator_dispatcher[LE.ExactlyExpression] = _dispatch_exactly
_operator_dispatcher[LE.AtLeastExpression] = _dispatch_atleast
_operator_dispatcher[LE.AtMostExpression] = _dispatch_atmost

_before_child_dispatcher = {}
_before_child_dispatcher[BV.ScalarBooleanVar] = _dispatch_boolean_var
_before_child_dispatcher[BV._GeneralBooleanVarData] = _dispatch_boolean_var
_before_child_dispatcher[AutoLinkedBooleanVar] = _dispatch_boolean_var
_before_child_dispatcher[ScalarVar] = _dispatch_var
_before_child_dispatcher[_GeneralVarData] = _dispatch_var
_before_child_dispatcher[_GeneralExpressionData] = _dispatch_expression
_before_child_dispatcher[ScalarExpression] = _dispatch_expression

class LogicalToLinearVisitor(StreamBasedExpressionVisitor):
    """Converts BooleanExpressions to Linear (MIP) representation

    This converter eschews conjunctive normal form, and instead follows
    the well-trodden MINLP path of factorable programming.

    """

    def __init__(self):
        super().__init__()
        self.z_vars = VarList(domain=Binary)
        self.constraints = ConstraintList()
        self.expansions = ComponentMap()
        self.boolean_to_binary_map = ComponentMap()

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        if child.__class__ in EXPR.native_types:
            return False, child

        if not child.is_expression_type() or child.is_named_expression_type():
            return _before_child_dispatcher[child.__class__](self, child)

        return True, None

    def exitNode(self, node, data):
        return _operator_dispatcher[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        # This LogicalExpression must evaluate to True
        result.fix(1)
        return result
