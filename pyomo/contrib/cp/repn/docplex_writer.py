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

import itertools

import pyomo.core.expr.current as EXPR

from pyomo.core.expr.visitor import (
    StreamBasedExpressionVisitor, identify_variables
)
from pyomo.core.base.set import SetProduct

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

class LogicalToDoCplex(StreamBasedExpressionVisitor):
    _operator_handles = {
        EXPR.GetItemExpression: _handle_getitem,
    }

    def __init__(self, cpx_model):
        self.cpx = cpx_model

    def beforeChild(self, node, child, child_idx):
        # Return native types
        if child.__class__ in native_types:
            return False, child

        # Convert Vars Logical vars to docplex equivalents
        # TODO

        return True, None

    def exitNode(self, node, data):
        return self._operator_handles[node.__class__](self, node, data)


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

