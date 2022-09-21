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

# TODO: How do we defer so this doesn't mess up everything?
import docplex.cp.model as cp

import itertools

from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.interval_var import (
    IntervalVarTimePoint, IntervalVarPresence, IntervalVarLength
)
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
    AtExpression, BeforeExpression
)
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    AlwaysIn
)

from pyomo.core.base.var import ScalarVar
import pyomo.core.expr.current as EXPR
from pyomo.core.expr.logical_expr import (
    AndExpression, OrExpression, XorExpression, NotExpression,
    EquivalenceExpression, ImplicationExpression, ExactlyExpression,
    AtMostExpression, AtLeastExpression
)
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.visitor import (
    StreamBasedExpressionVisitor, identify_variables
)
from pyomo.core.base.set import SetProduct

from pdb import set_trace

class _START_TIME(object): pass
class _END_TIME(object): pass
class _GENERAL(object): pass

def _check_var_domain(visitor, node, var):
    if not var.domain.isdiscrete():
        raise ValueError(
            "Variable indirection '%s' contains argument '%s', "
            "which is not a discrete variable" % (node, var))
    bnds = var.bounds
    if None in bnds:
        raise ValueError(
            "Variable indirection '%s' contains argument '%s', "
            "which is not restricted to a finite discrete domain"
            % (node, var))
    return var.domain & RangeSet(*bnds)

def _handle_getitem(visitor, node, data):
    # First we need to determine the range for each of the the
    # arguments.  They can be:
    #
    #  - simple values
    #  - docplex integer variables
    #  - docplex integer expressions
    arg_domain = []
    arg_scale = []
    expr = 0
    mult = 1
    # Note skipping the first argument: that should be the IndexedComponent
    for i, arg in enumerate(data[1:]):
        if arg.__class__ in native_types:
            arg_set = Set(initialize=[arg])
            arg_set.construct()
            arg_domain.append(arg_set)
            arg_scale.append(None)
        elif node.arg(i+1).is_expression_type(): #arg.is_docplex_expression():
            # This argument is an expression.  It could be any
            # combination of any number of integer variables, as long as
            # the resulting expression is still an IntExpression.  We
            # can't really rely on FBBT here, because we need to know
            # that the expression returns values in a regular domain
            # (i.e., the set of possible values has to have a start,
            # end, and finite, regular step).
            #
            # We will brute force it: go through every combination of
            # every variable and record the resulting expression value.
            arg_expr = node.arg(i+1)
            var_list = list(identify_variables(arg_expr, include_fixed=False))
            var_domain = [list(_check_var_domain(visitor, node, v))
                          for v in var_list]
            arg_vals = set()
            for var_vals in itertools.product(*var_domain):
                for v, val in zip(var_list, var_vals):
                    v.set_value(val)
                arg_vals.add(arg_expr())
            # Now that we have all the values that define the domain of
            # the result of the expression, stick them into a set and
            # rely on the Set infrastructure to calculate (and verify)
            # the interval.
            arg_set = Set(initialize=sorted(arg_vals))
            arg_set.construct()
            interval = arg_set.get_interval()
            if not interval[2]:
                raise ValueError(
                    "Variable indirection '%s' contains argument expression "
                    "'%s' that does not evaluate to a simple discrete set"
                    % (node, arg_expr))
            arg_domain.append(arg_set)
            arg_scale.append(interval)
        else:
            # This had better be a simple variable over a regular
            # discrete domain.  When we add support for ategorical
            # variables, we will need to ensure that the categoricals
            # have already been converted to simple integer domains by
            # this point.
            var = node.arg(i+1)
            arg_domain.append(_check_var_domain(visitor, node, var))
            arg_scale.append(arg_domain[-1].get_interval())
        # Buid the expression that maps arguments to GetItem() to a
        # position in the elements list
        if arg_scale[-1] is not None:
            _min, _max, _step = arg_scale[-1]
            expr += mult * (arg - _min) / _step
            # This could be (_max - _min) // _step + 1, but that assumes
            # tht the set correctly collapsed the bounds and that the
            # lower and upper bounds were part of the step.  That
            # *should* be the case for Set, but I am suffering from a
            # crisis of confidence at the moment.
            mult *= len(arg_domain[-1])
    # Get the list of all elements selectable by the argument
    # expression(s); fill in new variables for any indices allowable by
    # the argument expression(s) but not present in the IndexedComponent
    # indexing set.
    elements = []
    for idx in SetProduct(*arg_domain):
        try:
            elements.append(data[0][idx])
        except KeyError:
            # TODO: fill in bogus variable and add a constraint
            # disallowing it from being selected
            elements.append(None)
    return (elements, expr)

def _before_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            return False, child.value
        if child.domain in (Integers, PositiveIntegers, NonPositiveIntegers,
                            NegativeIntegers, NonNegativeIntegers):
            cpx_var = cp.integer_var(min=child.bounds[0], max=child.bounds[1])
        elif child.domain in (Binary, Boolean):
            # Sorry, universe, but docplex doesn't know the difference between
            # Boolean and Binary...
            cpx_var = cp.binary_var()
        else:
            raise ValueError("The LogicalToDoCplex writer can only support "
                             "integer- or Boolean-valued variables. Cannot "
                             "write Var %s with domain %s" % (child.name, 
                                                              child.domain))
        visitor.cpx.add(cpx_var)
        visitor.var_map[_id] = cpx_var
    return False, (_GENERAL, visitor.var_map[_id])

def _create_docplex_interval_var(interval_var):
    # Create a new docplex interval var and then figure out all the info that
    # gets stored on it
    cpx_interval_var = cp.interval_var()

    # Figure out if it exists
    if interval_var.is_present.fixed and not interval_var.is_present.value:
        # Someone has fixed that this will not get scheduled.
        cpx_interval_var.set_absent()
    elif interval_var.optional:
        cpx_interval_var.set_optional()
    else:
        cpx_interval_var.set_present()

    # Figure out constraints on its length
    length = interval_var.length.value if interval_var.length.fixed else \
             None
    if length is not None:
        cpx_interval_var.set_length(length)
    else:
        length = interval_var.length
        if length.lb is not None:
            cpx_interval_var.set_length_min(length.lb)
        if length.ub is not None:
            cpx_interval_var.set_length_max(length.ub)

    # Figure out constraints on start time
    start_time = interval_var.start_time
    start = start_time.value if start_time.fixed else None
    if start is not None:
        cpx_interval_var.set_start(start)
    else:
        if start_time.lb is not None:
            cpx_interval_var.set_start_min(start.lb)
        if start_time.ub is not None:
            cpx_interval_var.set_start_max(start.ub)

    # Figure out constraints on end time
    end_time = interval_var.end_time
    end = end_time.value if end_time.fixed else None
    if end is not None:
        cpx_interval_var.set_end(end)
    else:
        if end_time.lb is not None:
            cpx_interval_var.set_end_min(end.lb)
        if end_time.ub is not None:
            cpx_interval_var.set_end_max(end.ub)

    return cpx_interval_var

def _get_docplex_interval_var(interval_var):
    # We might already have the interval_var but be looking for a 
    # start_time or an end_time that we haven't looked for yet:
    if id(interval_var) in visitor.var_map:
        cpx_interval_var = visitor.var_map[id(interval_var)]
    else:
        cpx_interval_var = _create_docplex_interval_var(interval_var)
        visitor.cpx.add(cpx_interval_var)
    return cpx_interval_var
        
def _before_interval_var_time_point(visitor, child):
    _id = id(child)
    interval_var = child.get_associated_interval_var()
    if _id not in visitor.var_map:
        cpx_interval_var = _get_docplex_interval_var(interval_var)

        # Map the child to the right docplex expression
        if child.local_name == 'start_time':
            visitor.var_map[_id] = cp.start_of(cpx_interval_var)
            time_point = _START_TIME
        elif child.local_name == 'end_time':
            visitor.var_map[_id] = cp.end_of(cpx_interval_var)
            time_point = _END_TIME

    # We return the time point, the expression to get to it directly, and the
    # 'parent' interval var, because depending on the expression, we may want
    # the parent or the expression for the time point.
    return False, (time_point, (visitor.var_map[_id],
                                visitor.var_map[id(interval_var)]))

def _before_interval_var_length(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        interval_var = child.get_associated_interval_var()
        cpx_interval_var = _get_docplex_interval_var(interval_var)
        
        visitor.var_map[_id] = cp.length_of(cpx_interval_var)
    # There aren't any special types of constraints involving the length, so we
    # just treat this expression as if it's a normal variable.
    return False, (_GENERAL, visitor.var_map[_id])

def _before_interval_var_presence(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        interval_var = child.get_associated_interval_var()
        cpx_interval_var = _get_docplex_interval_var(interval_var)
        
        visitor.var_map[_id] = cp.presence_of(cpx_interval_var)
    # There aren't any special types of constraints involving the presence, so
    # we just treat this expression as if it's a normal variable.
    return False, visitor.var_map[_id]

##
# Algebraic expressions
##

def _handle_monomial_expr(visitor, node, arg1, arg2):
    return cp.times(arg1, arg2)

def _handle_sum_node(visitor, node, *args):
    return sum(args[1:], start=args[0])

def _handle_negation_node(visitor, node, arg1):
    return cp.times(-1, arg1)

def _handle_product_node(visitor, node, arg1, arg2):
    return cp.times(arg1, arg2)

def _handle_division_node(visitor, node, arg1, arg2):
    return cp.float_div(arg1, arg2)

def _handle_integer_division_node(visitor, node, arg1, arg2):
    return cp.int_div(arg1, arg2)

def _handle_pow_node(visitor, node, arg1, arg2):
    return cp.power(arg1, arg2)

def _handle_abs_node(visitor, node, arg1):
    return cp.abs(arg1)

def _handle_min_node(visitor, node, *args):
    return cp.min(args)

def _handle_max_node(visitor, node, *args):
    return cp.max(args)

##
# Logical expressions
##

def _handle_and_node(visitor, node, *args):
    return cp.logical_and(args)

def _handle_or_node(visitor, node, *args):
    return cp.logical_or(args)

def _handle_xor_node(visitor, node, arg1, arg2):
    return cp.equal(cp.count([arg1, arg2], True), 1)

def _handle_not_node(visitor, node, arg):
    return cp.logical_not(arg)

def _handle_equality_node(visitor, node, arg1, arg2):
    return cp.equal(arg1, arg2)

def _handle_equivalence_node(visitor, node, arg1, arg2):
    return cp.equal(arg1, arg2)

def _handle_inequality_node(visitor, node, arg1, arg2):
    return cp.less_or_equal(arg1, arg2)

def _handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    return (cp.less_or_equal(arg1, arg2), cp.less_or_equal(arg2, arg3))

def _handle_not_equal_node(visitor, node, arg1, arg2):
    return cp.diff(arg1, arg2)

def _handle_implication_node(visitor, node, arg1, arg2):
    return cp.if_then(arg1, arg2)

def _handle_exactly_node(visitor, node, *args):
    # TODO: if args[0] isn't a constant, then this is more complicated
    return cp.equal(cp.count(args[1:], True), args[0])

def _handle_at_most_node(visitor, node, *args):
    # TODO: if args[0] isn't a constant, then this is more complicated
    return cp.less_or_equal(cp.count(args[1:], True), args[0])

def _handle_at_least_node(visitor, node, *args):
    # TODO: if args[0] isn't a constant, then this is more complicated
    return cp.greater_or_equal(cp.count(args[1:], True), args[0])

##
# Scheduling
##


_precedence_exprs = {
    (_START_TIME, _START_TIME): cp.start_before_start,
    (_START_TIME, _END_TIME): cp.start_before_end,
    (_END_TIME, _START_TIME): cp.end_before_start,
    (_END_TIME, _END_TIME): cp.end_before_end,
}
def _handle_before_expression_node(visitor, node, before, after, delay):
    (first_interval, first_time_point) = before
    (second_interval, second_time_point) = after
    return _precedence_exprs[(first_time_point, second_time_point)](
        first_interval, second_interval)

def _handle_at_expression_node(visitor, node, arg1, arg2, delay):
    pass

def _handle_always_in_node(visitor, cumul_func, bounds, times):
    pass

class LogicalToDoCplex(StreamBasedExpressionVisitor):
    _operator_handles = {
        EXPR.GetItemExpression: _handle_getitem,
        EXPR.NegationExpression: _handle_negation_node,
        EXPR.ProductExpression: _handle_product_node,
        EXPR.DivisionExpression: _handle_division_node,
        EXPR.PowExpression: _handle_pow_node,
        EXPR.AbsExpression: _handle_abs_node,
        EXPR.MonomialTermExpression: _handle_monomial_expr,
        EXPR.SumExpression: _handle_sum_node,
        MinExpression: _handle_min_node,
        MaxExpression: _handle_max_node,
        NotExpression: _handle_not_node,
        EquivalenceExpression: _handle_equivalence_node,
        ImplicationExpression: _handle_implication_node,
        AndExpression: _handle_and_node,
        OrExpression: _handle_or_node,
        XorExpression: _handle_xor_node,
        ExactlyExpression: _handle_exactly_node,
        AtMostExpression: _handle_at_most_node,
        AtLeastExpression: _handle_at_least_node,
        EXPR.EqualityExpression: _handle_equality_node,
        EXPR.InequalityExpression: _handle_inequality_node,
        EXPR.RangedExpression: _handle_ranged_inequality_node,
        AtExpression: _handle_at_expression_node,
        BeforeExpression: _handle_before_expression_node,
        AlwaysIn: _handle_always_in_node,
    }
    _var_handles = {
        IntervalVarTimePoint: _before_interval_var_time_point,
        IntervalVarLength: _before_interval_var_length,
        IntervalVarPresence: _before_interval_var_presence,
        ScalarVar: _before_var
    }

    def __init__(self, cpx_model):
        self.cpx = cpx_model
        self._process_node = self._process_node_bx

        self.var_map = {}

    def initializeWalker(self, expr):
        expr, src, src_idx = expr
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, self.finalizeResult(result)
        return True, expr

    def beforeChild(self, node, child, child_idx):
        # Return native types
        if child.__class__ in native_types:
            return False, child

        # Convert Vars Logical vars to docplex equivalents
        # TODO
        if not child.is_expression_type():
            if child.is_potentially_variable():
                return self._var_handles[child.__class__](self, child)
                #return _before_var(self, child)
            else:
                raise NotImplementedError()

        return True, None

    def exitNode(self, node, data):
        print("EXIT\n\tnode: %s\n\tdata: %s" % (node, data))
        return self._operator_handles[node.__class__](self, node, *data)

    finalizeResult = None

    def _declare_docplex_algebraic_var(self, var):
        pass

    def _declare_docplex_interval_var(self, var):
        pass


if __name__ == '__main__':
    from pyomo.common.formatting import tostr
    from pyomo.environ import *
    m = ConcreteModel()
    m.I = RangeSet(10)
    m.a = Var(m.I)
    m.x = Var(within=PositiveIntegers, bounds=(6,8))

    e = m.a[m.x]
    ans = _handle_getitem(None, e, [m.a, m.x])
    print("\n", e)
    print(tostr(ans))

    m.b = Var(m.I, m.I)
    m.y = Var(within=[1, 3, 5])

    e = m.b[m.x, 3]
    ans = _handle_getitem(None, e, [m.b, m.x, 3])
    print("\n", e)
    print(tostr(ans))

    e = m.b[3, m.x]
    ans = _handle_getitem(None, e, [m.b, 3, m.x])
    print("\n", e)
    print(tostr(ans))

    e = m.b[m.x, m.x]
    ans = _handle_getitem(None, e, [m.b, m.x, m.x])
    print("\n", e)
    print(tostr(ans))

    e = m.b[m.x, m.y]
    ans = _handle_getitem(None, e, [m.b, m.x, m.y])
    print("\n", e)
    print(tostr(ans))

    e = m.a[m.x - m.y]
    ans = _handle_getitem(None, e, [m.a, m.x - m.y])
    print("\n", e)
    print(tostr(ans))

    docplex_model= cp.CpoModel()
    visitor = LogicalToDoCplex(docplex_model)

    m.c = Constraint(expr=m.x**2 + 4 + 2*6*m.x/(4*m.x) <= 3)
    expr = visitor.walk_expression((m.c.body, m.c, 0))
    print(expr)

    m.i = IntervalVar(optional=True)
    m.i2 = IntervalVar([1, 2], optional=False)
    m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[1].end_time))
    expr = visitor.walk_expression((m.c.body, m.c, 0))
    print(expr)
