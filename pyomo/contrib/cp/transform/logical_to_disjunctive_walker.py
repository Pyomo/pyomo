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

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
import pyomo.core.expr as EXPR
from pyomo.core.base import (
    Binary,
    Constraint,
    ConstraintList,
    NonNegativeIntegers,
    VarList,
    value,
)
import pyomo.core.base.boolean_var as BV
from pyomo.core.base.expression import ScalarExpression, ExpressionData
from pyomo.core.base.param import ScalarParam, ParamData
from pyomo.core.base.var import ScalarVar, VarData
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, Disjunct, Disjunction


def _dispatch_boolean_const(visitor, node):
    return False, 1 if node.value else 0


def _dispatch_boolean_var(visitor, node):
    if node not in visitor.boolean_to_binary_map:
        binary = node.get_associated_binary()
        if binary is not None:
            visitor.boolean_to_binary_map[node] = binary
        else:
            z = visitor.z_vars.add()
            visitor.boolean_to_binary_map[node] = z
            node.associate_binary_var(z)
    if node.fixed:
        visitor.boolean_to_binary_map[node].fixed = True
        visitor.boolean_to_binary_map[node].set_value(
            int(node.value) if node.value is not None else None, skip_validation=True
        )
    return False, visitor.boolean_to_binary_map[node]


def _dispatch_var(visitor, node):
    return False, node


def _dispatch_param(visitor, node):
    return False, node


def _dispatch_expression(visitor, node):
    return False, node.expr


def _before_relational_expr(visitor, node):
    raise MouseTrap(
        "The RelationalExpression '%s' was used as a Boolean term "
        "in a logical proposition. This is not yet supported "
        "when transforming to disjunctive form." % node
    )


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
        visitor,
        node,
        _dispatch_or(visitor, node, 1 - a, b),
        _dispatch_or(visitor, node, a, 1 - b),
    )


def _dispatch_and(visitor, node, *args):
    # z == a ^ b ^ ...
    z = visitor.z_vars.add()
    for arg in args:
        visitor.constraints.add(arg >= z)
    visitor.constraints.add(len(args) - sum(args) >= 1 - z)
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


def _get_integer_value(n, node):
    if n.__class__ in EXPR.native_numeric_types and int(n) == n:
        return n
    if n.__class__ not in EXPR.native_types:
        if n.is_potentially_variable():
            # [ESJ 11/22]: This is probably worth supporting sometime, but right
            # now we are abiding by what docplex allows in their 'count'
            # function. Part of supporting this will be making sure we catch
            # strict inequalities in the GDP transformations. Because if we
            # don't know that n is integer-valued we will be forced to write
            # strict inequalities instead of incrememting or decrementing by 1
            # in the disjunctions.
            raise MouseTrap(
                "The first argument '%s' to '%s' is potentially variable. "
                "This may be a mathematically coherent expression; However "
                "it is not yet supported to convert it to a disjunctive "
                "program." % (n, node)
            )
        else:
            return n
    raise ValueError(
        "The first argument to '%s' must be an integer.\n\tRecieved: %s" % (node, n)
    )


def _dispatch_exactly(visitor, node, *args):
    # z = sum(args[1:]) == args[0]
    # This is currently implemented as:
    # [sum(args[1:] = n] v [[sum(args[1:]) < n] v [sum(args[1:]) > n]]
    M = len(args) - 1
    n = _get_integer_value(args[0], node)
    sum_expr = sum(args[1:])
    equality_disj = visitor.disjuncts[len(visitor.disjuncts)]
    equality_disj.constraint = Constraint(expr=sum_expr == n)
    inequality_disj = visitor.disjuncts[len(visitor.disjuncts)]
    inequality_disj.disjunction = Disjunction(
        expr=[[sum_expr <= n - 1], [sum_expr >= n + 1]]
    )
    visitor.disjunctions[len(visitor.disjunctions)] = [equality_disj, inequality_disj]
    return equality_disj.indicator_var.get_associated_binary()


def _dispatch_atleast(visitor, node, *args):
    # z = sum[args[1:] >= n
    # This is implemented as:
    # [sum(args[1:] >= n] v [sum(args[1:] < n]
    n = _get_integer_value(args[0], node)
    sum_expr = sum(args[1:])
    atleast_disj = visitor.disjuncts[len(visitor.disjuncts)]
    less_disj = visitor.disjuncts[len(visitor.disjuncts)]
    atleast_disj.constraint = Constraint(expr=sum_expr >= n)
    less_disj.constraint = Constraint(expr=sum_expr <= n - 1)
    visitor.disjunctions[len(visitor.disjunctions)] = [atleast_disj, less_disj]
    return atleast_disj.indicator_var.get_associated_binary()


def _dispatch_atmost(visitor, node, *args):
    # z = sum[args[1:] <= n
    # This is implemented as:
    # [sum(args[1:] <= n] v [sum(args[1:] > n]
    n = _get_integer_value(args[0], node)
    sum_expr = sum(args[1:])
    atmost_disj = visitor.disjuncts[len(visitor.disjuncts)]
    more_disj = visitor.disjuncts[len(visitor.disjuncts)]
    atmost_disj.constraint = Constraint(expr=sum_expr <= n)
    more_disj.constraint = Constraint(expr=sum_expr >= n + 1)
    visitor.disjunctions[len(visitor.disjunctions)] = [atmost_disj, more_disj]
    return atmost_disj.indicator_var.get_associated_binary()


_operator_dispatcher = {}
_operator_dispatcher[EXPR.ImplicationExpression] = _dispatch_implication
_operator_dispatcher[EXPR.EquivalenceExpression] = _dispatch_equivalence
_operator_dispatcher[EXPR.NotExpression] = _dispatch_not
_operator_dispatcher[EXPR.AndExpression] = _dispatch_and
_operator_dispatcher[EXPR.OrExpression] = _dispatch_or
_operator_dispatcher[EXPR.XorExpression] = _dispatch_xor
_operator_dispatcher[EXPR.ExactlyExpression] = _dispatch_exactly
_operator_dispatcher[EXPR.AtLeastExpression] = _dispatch_atleast
_operator_dispatcher[EXPR.AtMostExpression] = _dispatch_atmost

_before_child_dispatcher = {}
_before_child_dispatcher[EXPR.BooleanConstant] = _dispatch_boolean_const
_before_child_dispatcher[BV.ScalarBooleanVar] = _dispatch_boolean_var
_before_child_dispatcher[BV.BooleanVarData] = _dispatch_boolean_var
_before_child_dispatcher[AutoLinkedBooleanVar] = _dispatch_boolean_var
_before_child_dispatcher[ParamData] = _dispatch_param
_before_child_dispatcher[ScalarParam] = _dispatch_param
# for the moment, these are all just so we can get good error messages when we
# don't handle them:
_before_child_dispatcher[ScalarVar] = _dispatch_var
_before_child_dispatcher[VarData] = _dispatch_var
_before_child_dispatcher[ExpressionData] = _dispatch_expression
_before_child_dispatcher[ScalarExpression] = _dispatch_expression


class LogicalToDisjunctiveVisitor(StreamBasedExpressionVisitor):
    """Converts BooleanExpressions to Linear (MIP) representation

    This converter eschews conjunctive normal form, and instead follows
    the well-trodden MINLP path of factorable programming.

    """

    def __init__(self):
        super().__init__()
        self.z_vars = VarList(domain=Binary)
        self.z_vars.construct()
        self.constraints = ConstraintList()
        self.disjuncts = Disjunct(NonNegativeIntegers, concrete=True)
        self.disjunctions = Disjunction(NonNegativeIntegers)
        self.disjunctions.construct()
        self.expansions = ComponentMap()
        self.boolean_to_binary_map = ComponentMap()

    def initializeWalker(self, expr):
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        if child.__class__ in EXPR.native_types:
            if child.__class__ is bool:
                # If we encounter a bool, we are going to need to treat it as
                # binary explicitly because we are finally pedantic enough in the
                # expression system to not allow some of the mixing we will need
                # (like summing a LinearExpression with a bool)
                return False, int(child)
            return False, child

        if child.is_numeric_type():
            # Just pass it through, we'll figure it out later
            return False, child
        if child.is_expression_type(ExpressionType.RELATIONAL):
            # Eventually we'll handle these. Right now we set a MouseTrap
            return _before_relational_expr(self, child)

        if not child.is_expression_type() or child.is_named_expression_type():
            return _before_child_dispatcher[child.__class__](self, child)

        return True, None

    def exitNode(self, node, data):
        return _operator_dispatcher[node.__class__](self, node, *data)

    def finalizeResult(self, result):
        # This LogicalExpression must evaluate to True (but note that we cannot
        # fix this variable to 1 since this logical expression could be living
        # on a Disjunct and later need to be relaxed.)
        expr = result >= 1
        if expr.__class__ is bool:
            self.constraints.add(Constraint.Feasible if expr else Constraint.Infeasible)
        else:
            self.constraints.add(expr)
        return result
