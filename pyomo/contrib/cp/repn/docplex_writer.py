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

from pyomo.common.dependencies import attempt_import

import itertools
import logging
from operator import attrgetter

from pyomo.common import DeveloperError
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.collections import ComponentMap
from pyomo.common.fileutils import Executable

from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.interval_var import (
    IntervalVarStartTime,
    IntervalVarEndTime,
    IntervalVarPresence,
    IntervalVarLength,
    ScalarIntervalVar,
    IntervalVarData,
    IndexedIntervalVar,
)
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
    BeforeExpression,
    AtExpression,
)
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
    AlwaysIn,
    StepAt,
    StepAtStart,
    StepAtEnd,
    Pulse,
    CumulativeFunction,
    NegatedStepFunction,
)

from pyomo.core.base import (
    minimize,
    maximize,
    SortComponents,
    Block,
    Objective,
    Constraint,
    Var,
    Param,
    BooleanVar,
    LogicalConstraint,
    Suffix,
    value,
)
from pyomo.core.base.boolean_var import (
    ScalarBooleanVar,
    _GeneralBooleanVarData,
    IndexedBooleanVar,
)
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.param import IndexedParam, ScalarParam
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
import pyomo.core.expr as EXPR
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.core.base import Set, RangeSet
from pyomo.core.base.set import SetProduct
from pyomo.opt import WriterFactory, SolverFactory, TerminationCondition, SolverResults

### FIXME: Remove the following as soon as non-active components no
### longer report active==True
from pyomo.network import Port

###


def _finalize_docplex(module, available):
    if not available:
        return
    _deferred_element_getattr_dispatcher['start_time'] = module.start_of
    _deferred_element_getattr_dispatcher['end_time'] = module.end_of
    _deferred_element_getattr_dispatcher['length'] = module.length_of
    _deferred_element_getattr_dispatcher['is_present'] = module.presence_of

    # Scheduling dispatchers
    _before_dispatchers[_START_TIME, _START_TIME] = module.start_before_start
    _before_dispatchers[_START_TIME, _END_TIME] = module.start_before_end
    _before_dispatchers[_END_TIME, _START_TIME] = module.end_before_start
    _before_dispatchers[_END_TIME, _END_TIME] = module.end_before_end

    _at_dispatchers[_START_TIME, _START_TIME] = module.start_at_start
    _at_dispatchers[_START_TIME, _END_TIME] = module.start_at_end
    _at_dispatchers[_END_TIME, _START_TIME] = module.end_at_start
    _at_dispatchers[_END_TIME, _END_TIME] = module.end_at_end

    _time_point_dispatchers[_START_TIME] = module.start_of
    _time_point_dispatchers[_END_TIME] = module.end_of


cp, docplex_available = attempt_import('docplex.cp.model', callback=_finalize_docplex)
cp_solver, docplex_available = attempt_import('docplex.cp.solver')

logger = logging.getLogger('pyomo.contrib.cp')


# These are things that don't need special handling:
class _GENERAL(object):
    pass


# These are operations that need to be deferred sometimes, usually because of
# indirection:
class _START_TIME(object):
    pass


class _END_TIME(object):
    pass


class _DEFERRED_ELEMENT_CONSTRAINT(object):
    pass


class _ELEMENT_CONSTRAINT(object):
    pass


class _DEFERRED_BEFORE(object):
    pass


class _DEFERRED_AFTER(object):
    pass


class _DEFERRED_AT(object):
    pass


class _BEFORE(object):
    pass


class _AT(object):
    pass


class _IMPLIES(object):
    pass


class _LAND(object):
    pass


class _LOR(object):
    pass


class _XOR(object):
    pass


class _EQUIVALENT_TO(object):
    pass


def _check_var_domain(visitor, node, var):
    if not var.domain.isdiscrete():
        # Note: in the context of the current writer, this should be unreachable
        # because we can't handle non-discrete variables at all, so there will
        # already be errors handling the children of this expression.
        raise ValueError(
            "Variable indirection '%s' contains argument '%s', "
            "which is not a discrete variable" % (node, var)
        )
    bnds = var.bounds
    if None in bnds:
        raise ValueError(
            "Variable indirection '%s' contains argument '%s', "
            "which is not restricted to a finite discrete domain" % (node, var)
        )
    return var.domain & RangeSet(*bnds)


def _handle_getitem(visitor, node, *data):
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
    # Note: skipping the first argument: that should be the IndexedComponent
    for i, arg in enumerate(data[1:]):
        if arg[1].__class__ in EXPR.native_types:
            arg_set = Set(initialize=[arg[1]])
            arg_set.construct()
            arg_domain.append(arg_set)
            arg_scale.append(None)
        elif node.arg(i + 1).is_expression_type():
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
            arg_expr = node.arg(i + 1)
            var_list = list(identify_variables(arg_expr, include_fixed=False))
            var_domain = [list(_check_var_domain(visitor, node, v)) for v in var_list]
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
                    % (node, arg_expr)
                )
            arg_domain.append(arg_set)
            arg_scale.append(interval)
        else:
            # This had better be a simple variable over a regular
            # discrete domain.  When we add support for categorical
            # variables, we will need to ensure that the categoricals
            # have already been converted to simple integer domains by
            # this point.
            var = node.arg(i + 1)
            arg_domain.append(_check_var_domain(visitor, node, var))
            arg_scale.append(arg_domain[-1].get_interval())
        # Build the expression that maps arguments to GetItem() to a
        # position in the elements list
        if arg_scale[-1] is not None:
            _min, _max, _step = arg_scale[-1]
            # ESJ: Have to use integer division here because otherwise, later,
            # when we construct the element constraint, docplex won't believe
            # the index is an integer expression.
            if _step is None:
                raise ValueError(
                    "Variable indirection '%s' is over a discrete domain "
                    "without a constant step size. This is not supported." % node
                )
            expr += mult * (arg[1] - _min) // _step
            # This could be (_max - _min) // _step + 1, but that assumes
            # that the set correctly collapsed the bounds and that the
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
            idx = idx if len(idx) > 1 else idx[0]
            elements.append(data[0][1][idx])
        except KeyError:
            raise ValueError(
                "Variable indirection '%s' permits an index '%s' "
                "that is not a valid key. In CP Optimizer, this is a "
                "structural infeasibility." % (node, idx)
            )
            # NOTE: If we thought it was the right thing to do in the future, we
            # could fill in with a bogus variable and add a constraint
            # disallowing it from being selected
    try:
        return (_ELEMENT_CONSTRAINT, cp.element(elements, expr))
    except AssertionError:
        return (_DEFERRED_ELEMENT_CONSTRAINT, (elements, expr))


_element_constraint_attr_dispatcher = {
    'before': _DEFERRED_BEFORE,
    'after': _DEFERRED_AFTER,
    'at': _DEFERRED_AT,
    'implies': _IMPLIES,
    'land': _LAND,
    'lor': _LOR,
    'xor': _XOR,
    'equivalent_to': _EQUIVALENT_TO,
}
# This will get populated when cp is finally imported
_deferred_element_getattr_dispatcher = {}


def _handle_getattr(visitor, node, obj, attr):
    # We either end up here because we do not yet know the list of variables to
    # make an element constraint (the first case) or because we are asking for
    # an attribute on something with indirection, so at this point we *have* a
    # constructed element constraint (the second case).
    if obj[0] is _DEFERRED_ELEMENT_CONSTRAINT:
        # then obj[1] is a list of cp thingies that we need to get the attr on,
        # and then at the end we need to make the element constraint we couldn't
        # make before.
        try:
            ans = list(map(_deferred_element_getattr_dispatcher[attr[1]], obj[1][0]))
        except KeyError:
            logger.error("Unrecognized attribute in GetAttrExpression: %s." % attr[1])
            raise
        return (_ELEMENT_CONSTRAINT, cp.element(array=ans, index=obj[1][1]))
    elif obj[0] is _ELEMENT_CONSTRAINT:
        try:
            return (_element_constraint_attr_dispatcher[attr[1]], obj)
        except KeyError:
            logger.error(
                "Unrecognized attribute in GetAttrExpression:"
                "%s. Found for object: %s" % (attr[1], obj[1])
            )
            raise
    else:
        raise DeveloperError(
            "Unrecognized argument type '%s' to getattr dispatcher." % obj[0]
        )


def _before_boolean_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            return False, (_GENERAL, child.value)
        nm = child.name if visitor.symbolic_solver_labels else None
        # Sorry, universe, but docplex doesn't know the difference between
        # Boolean and Binary...
        cpx_var = cp.binary_var(name=nm)
        # Because I want to pretend the world is sane from here on out, we will
        # return a Boolean expression (in docplex land) so this can be used as
        # an argument to logical expressions later
        visitor.var_map[_id] = cpx_var == 1
        visitor.pyomo_to_docplex[child] = cpx_var
    return False, (_GENERAL, visitor.var_map[_id])


def _before_indexed_boolean_var(visitor, child):
    cpx_vars = {}
    for i, v in child.items():
        if v.fixed:
            cpx_vars[i] = v.value
            continue
        cpx_var = cp.binary_var(name=v.name if visitor.symbolic_solver_labels else None)
        visitor.cpx.add(cpx_var)
        visitor.var_map[id(v)] = cpx_var == 1
        visitor.pyomo_to_docplex[v] = cpx_var
        cpx_vars[i] = cpx_var == 1
    return False, (_GENERAL, cpx_vars)


def _before_param(visitor, child):
    return False, (_GENERAL, value(child))


def _before_indexed_param(visitor, child):
    return False, (_GENERAL, {idx: value(p) for idx, p in child.items()})


def _create_docplex_var(pyomo_var, name=None):
    if pyomo_var.is_binary():
        return cp.binary_var(name=name)
    elif pyomo_var.is_integer():
        return cp.integer_var(
            min=pyomo_var.bounds[0], max=pyomo_var.bounds[1], name=name
        )
    elif pyomo_var.domain.isdiscrete():
        if pyomo_var.domain.isfinite():
            return cp.integer_var(domain=[d for d in pyomo_var.domain], name=name)
        else:
            # If we ever want to handle this case, I think we might be able to
            # make a normal integer var and then constrain it into the
            # domain. But no reason to go to the effort for now because I don't
            # know if the solver can even work with such a var.
            raise ValueError(
                "The LogicalToDoCplex writer does not support "
                "infinite discrete domains. Cannot write "
                "Var '%s' with domain '%s'" % (pyomo_var.name, pyomo_var.domain)
            )
    else:
        raise ValueError(
            "The LogicalToDoCplex writer can only support "
            "integer- or Boolean-valued variables. Cannot "
            "write Var '%s' with domain '%s'" % (pyomo_var.name, pyomo_var.domain)
        )


def _before_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        if child.fixed:
            return False, (_GENERAL, child.value)
        cpx_var = _create_docplex_var(
            child, name=child.name if visitor.symbolic_solver_labels else None
        )
        visitor.cpx.add(cpx_var)
        visitor.var_map[_id] = cpx_var
        visitor.pyomo_to_docplex[child] = cpx_var
    return False, (_GENERAL, visitor.var_map[_id])


def _before_indexed_var(visitor, child):
    cpx_vars = {}
    for i, v in child.items():
        cpx_var = _create_docplex_var(
            v, name=v.name if visitor.symbolic_solver_labels else None
        )
        visitor.cpx.add(cpx_var)
        visitor.var_map[id(v)] = cpx_var
        visitor.pyomo_to_docplex[v] = cpx_var
        cpx_vars[i] = cpx_var
    return False, (_GENERAL, cpx_vars)


def _handle_named_expression_node(visitor, node, expr):
    visitor._named_expressions[id(node)] = expr[1]
    return expr


def _before_named_expression(visitor, child):
    _id = id(child)
    if _id not in visitor._named_expressions:
        return True, None
    return False, (_GENERAL, visitor._named_expressions[_id])


def _create_docplex_interval_var(visitor, interval_var):
    # Create a new docplex interval var and then figure out all the info that
    # gets stored on it
    nm = interval_var.name if visitor.symbolic_solver_labels else None
    cpx_interval_var = cp.interval_var(name=nm)
    visitor.var_map[id(interval_var)] = cpx_interval_var

    # Figure out if it exists
    if interval_var.is_present.fixed and not interval_var.is_present.value:
        # Someone has fixed that this will not get scheduled.
        cpx_interval_var.set_absent()
    elif interval_var.optional:
        cpx_interval_var.set_optional()
    else:
        cpx_interval_var.set_present()

    # Figure out constraints on its length
    length = interval_var.length
    if length.fixed:
        cpx_interval_var.set_length(length.value)
    if length.lb is not None:
        cpx_interval_var.set_length_min(length.lb)
    if length.ub is not None:
        cpx_interval_var.set_length_max(length.ub)

    # Figure out constraints on start time
    start_time = interval_var.start_time
    if start_time.fixed:
        cpx_interval_var.set_start(start_time.value)
    else:
        if start_time.lb is not None:
            cpx_interval_var.set_start_min(start_time.lb)
        if start_time.ub is not None:
            cpx_interval_var.set_start_max(start_time.ub)

    # Figure out constraints on end time
    end_time = interval_var.end_time
    if end_time.fixed:
        cpx_interval_var.set_end(end_time.value)
    else:
        if end_time.lb is not None:
            cpx_interval_var.set_end_min(end_time.lb)
        if end_time.ub is not None:
            cpx_interval_var.set_end_max(end_time.ub)

    return cpx_interval_var


def _get_docplex_interval_var(visitor, interval_var):
    # We might already have the interval_var and just need to retrieve it
    if id(interval_var) in visitor.var_map:
        cpx_interval_var = visitor.var_map[id(interval_var)]
    else:
        cpx_interval_var = _create_docplex_interval_var(visitor, interval_var)
        visitor.cpx.add(cpx_interval_var)
    return cpx_interval_var


def _before_interval_var(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        cpx_interval_var = _get_docplex_interval_var(visitor, child)
        visitor.var_map[_id] = cpx_interval_var
        visitor.pyomo_to_docplex[child] = cpx_interval_var

    return False, (_GENERAL, visitor.var_map[_id])


def _before_indexed_interval_var(visitor, child):
    cpx_vars = {}
    for i, v in child.items():
        cpx_interval_var = _get_docplex_interval_var(visitor, v)
        visitor.var_map[id(v)] = cpx_interval_var
        visitor.pyomo_to_docplex[v] = cpx_interval_var
        cpx_vars[i] = cpx_interval_var
    return False, (_GENERAL, cpx_vars)


def _before_interval_var_start_time(visitor, child):
    _id = id(child)
    interval_var = child.get_associated_interval_var()
    if _id not in visitor.var_map:
        cpx_interval_var = _get_docplex_interval_var(visitor, interval_var)

    return False, (_START_TIME, visitor.var_map[id(interval_var)])


def _before_interval_var_end_time(visitor, child):
    _id = id(child)
    interval_var = child.get_associated_interval_var()
    if _id not in visitor.var_map:
        cpx_interval_var = _get_docplex_interval_var(visitor, interval_var)

    return False, (_END_TIME, visitor.var_map[id(interval_var)])


def _before_interval_var_length(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        interval_var = child.get_associated_interval_var()
        cpx_interval_var = _get_docplex_interval_var(visitor, interval_var)

        visitor.var_map[_id] = cp.length_of(cpx_interval_var)
    # There aren't any special types of constraints involving the length, so we
    # just treat this expression as if it's a normal variable.
    return False, (_GENERAL, visitor.var_map[_id])


def _before_interval_var_presence(visitor, child):
    _id = id(child)
    if _id not in visitor.var_map:
        interval_var = child.get_associated_interval_var()
        cpx_interval_var = _get_docplex_interval_var(visitor, interval_var)

        visitor.var_map[_id] = cp.presence_of(cpx_interval_var)
    # There aren't any special types of constraints involving the presence, so
    # we just treat this expression as if it's a normal variable.
    return False, (_GENERAL, visitor.var_map[_id])


def _handle_step_at_node(visitor, node):
    return cp.step_at(node._time, node._height)


def _handle_step_at_start_node(visitor, node):
    cpx_var = _get_docplex_interval_var(visitor, node._time)
    return cp.step_at_start(cpx_var, node._height)


def _handle_step_at_end_node(visitor, node):
    cpx_var = _get_docplex_interval_var(visitor, node._time)
    return cp.step_at_end(cpx_var, node._height)


def _handle_pulse_node(visitor, node):
    cpx_var = _get_docplex_interval_var(visitor, node._interval_var)
    return cp.pulse(cpx_var, node._height)


def _handle_negated_step_function_node(visitor, node):
    return _step_function_handles[node.args[0].__class__](visitor, node.args[0])


def _handle_cumulative_function(visitor, node):
    expr = 0
    for arg in node.args:
        if arg.__class__ is NegatedStepFunction:
            expr -= _handle_negated_step_function_node(visitor, arg)
        else:
            expr += _step_function_handles[arg.__class__](visitor, arg)

    return False, (_GENERAL, expr)


_step_function_handles = {
    StepAt: _handle_step_at_node,
    StepAtStart: _handle_step_at_start_node,
    StepAtEnd: _handle_step_at_end_node,
    Pulse: _handle_pulse_node,
    CumulativeFunction: _handle_cumulative_function,
    NegatedStepFunction: _handle_negated_step_function_node,
}
step_func_expression_types = _step_function_handles.keys()

##
# Algebraic expressions
##


def _get_int_valued_expr(arg):
    if arg[0] in {_GENERAL, _ELEMENT_CONSTRAINT}:
        return arg[1]
    elif arg[0] is _START_TIME:
        return cp.start_of(arg[1])
    elif arg[0] is _END_TIME:
        return cp.end_of(arg[1])
    else:
        raise DeveloperError(
            "Attempting to get a docplex integer-valued "
            "expression from object in class %s" % str(arg[0])
        )


def _get_bool_valued_expr(arg):
    if arg[0] is _GENERAL:
        return arg[1]
    elif arg[0] is _ELEMENT_CONSTRAINT:
        # docplex doesn't bother to check if 'element' expressions are integer-
        # or boolean-valued: they just complain if you use them in a boolean
        # context. So if we are about to use one that way, we set it equivalent
        # to True so that it will be boolean-valued according to docplex's
        # idiosyncrasies.
        return arg[1] == True
    elif arg[0] is _BEFORE:
        # We're using a start-before-start or its ilk in a boolean-valued
        # context. docplex doesn't believe these things are boolean-valued, so
        # we have to convert to the inequality version:
        (lhs, rhs) = arg[2]
        return _handle_inequality_node(None, None, lhs, rhs)[1]
    elif arg[0] is _AT:
        # Same as above, but now we need an equality node
        (lhs, rhs) = arg[2]
        return _handle_equality_node(None, None, lhs, rhs)[1]
    else:
        raise DeveloperError(
            "Attempting to get a docplex Boolean-valued "
            "expression from object in class %s" % str(arg[0])
        )


def _handle_monomial_expr(visitor, node, arg1, arg2):
    # Monomial terms show up a lot.  This handles some common
    # simplifications (necessary in part for the unit tests)
    if arg2[1].__class__ in EXPR.native_types:
        return _GENERAL, arg1[1] * arg2[1]
    elif arg1[1] == 1:
        return arg2
    return (_GENERAL, cp.times(_get_int_valued_expr(arg1), _get_int_valued_expr(arg2)))


def _handle_sum_node(visitor, node, *args):
    return (
        _GENERAL,
        sum(
            (_get_int_valued_expr(arg) for arg in args[1:]),
            _get_int_valued_expr(args[0]),
        ),
    )


def _handle_negation_node(visitor, node, arg1):
    return (_GENERAL, cp.times(-1, _get_int_valued_expr(arg1)))


def _handle_product_node(visitor, node, arg1, arg2):
    return (_GENERAL, cp.times(_get_int_valued_expr(arg1), _get_int_valued_expr(arg2)))


def _handle_division_node(visitor, node, arg1, arg2):
    return (
        _GENERAL,
        cp.float_div(_get_int_valued_expr(arg1), _get_int_valued_expr(arg2)),
    )


def _handle_pow_node(visitor, node, arg1, arg2):
    return (_GENERAL, cp.power(_get_int_valued_expr(arg1), _get_int_valued_expr(arg2)))


def _handle_abs_node(visitor, node, arg1):
    return (_GENERAL, cp.abs(_get_int_valued_expr(arg1)))


def _handle_min_node(visitor, node, *args):
    return (_GENERAL, cp.min((_get_int_valued_expr(arg) for arg in args)))


def _handle_max_node(visitor, node, *args):
    return (_GENERAL, cp.max((_get_int_valued_expr(arg) for arg in args)))


##
# Relational expressions
##


def _handle_equality_node(visitor, node, arg1, arg2):
    return (_GENERAL, cp.equal(_get_int_valued_expr(arg1), _get_int_valued_expr(arg2)))


def _handle_inequality_node(visitor, node, arg1, arg2):
    return (
        _GENERAL,
        cp.less_or_equal(_get_int_valued_expr(arg1), _get_int_valued_expr(arg2)),
    )


def _handle_ranged_inequality_node(visitor, node, arg1, arg2, arg3):
    return (
        _GENERAL,
        cp.range(
            _get_int_valued_expr(arg2),
            lb=_get_int_valued_expr(arg1),
            ub=_get_int_valued_expr(arg3),
        ),
    )


def _handle_not_equal_node(visitor, node, arg1, arg2):
    return (_GENERAL, cp.diff(_get_int_valued_expr(arg1), _get_int_valued_expr(arg2)))


##
# Logical expressions
##


def _handle_and_node(visitor, node, *args):
    return (_GENERAL, cp.logical_and((_get_bool_valued_expr(arg) for arg in args)))


def _handle_or_node(visitor, node, *args):
    return (_GENERAL, cp.logical_or((_get_bool_valued_expr(arg) for arg in args)))


def _handle_xor_node(visitor, node, arg1, arg2):
    return (
        _GENERAL,
        cp.equal(
            cp.count([_get_bool_valued_expr(arg1), _get_bool_valued_expr(arg2)], 1), 1
        ),
    )


def _handle_not_node(visitor, node, arg):
    return (_GENERAL, cp.logical_not(_get_bool_valued_expr(arg)))


def _handle_equivalence_node(visitor, node, arg1, arg2):
    return (
        _GENERAL,
        cp.equal(_get_bool_valued_expr(arg1), _get_bool_valued_expr(arg2)),
    )


def _handle_implication_node(visitor, node, arg1, arg2):
    return (
        _GENERAL,
        cp.if_then(_get_bool_valued_expr(arg1), _get_bool_valued_expr(arg2)),
    )


def _handle_exactly_node(visitor, node, *args):
    return (
        _GENERAL,
        cp.equal(
            cp.count((_get_bool_valued_expr(arg) for arg in args[1:]), 1),
            _get_int_valued_expr(args[0]),
        ),
    )


def _handle_at_most_node(visitor, node, *args):
    return (
        _GENERAL,
        cp.less_or_equal(
            cp.count((_get_bool_valued_expr(arg) for arg in args[1:]), 1),
            _get_int_valued_expr(args[0]),
        ),
    )


def _handle_at_least_node(visitor, node, *args):
    return (
        _GENERAL,
        cp.greater_or_equal(
            cp.count((_get_bool_valued_expr(arg) for arg in args[1:]), 1),
            _get_int_valued_expr(args[0]),
        ),
    )


## CallExpression handllers


def _before_call_dispatcher(visitor, node, *args):
    if len(args) == 2:
        return _handle_inequality_node(visitor, node, args[0], args[1])
    else:  # a delay is also specified
        lhs = _handle_sum_node(visitor, node, args[0], args[2])
        return _handle_inequality_node(visitor, node, lhs, args[1])


def _after_call_dispatcher(visitor, node, *args):
    if len(args) == 2:
        return _handle_inequality_node(visitor, node, args[1], args[0])
    else:  # delay is also specified
        lhs = _handle_sum_node(visitor, node, args[1], args[2])
        return _handle_inequality_node(visitor, node, lhs, args[0])


def _at_call_dispatcher(visitor, node, *args):
    if len(args) == 2:
        return _handle_equality_node(visitor, node, args[0], args[1])
    else:  # a delay is also specified
        rhs = _handle_sum_node(visitor, node, args[1], args[2])
        return _handle_equality_node(visitor, node, args[0], rhs)


_call_dispatchers = {
    _DEFERRED_BEFORE: _before_call_dispatcher,
    _DEFERRED_AFTER: _after_call_dispatcher,
    _DEFERRED_AT: _at_call_dispatcher,
    _IMPLIES: _handle_implication_node,
    _LAND: _handle_and_node,
    _LOR: _handle_or_node,
    _XOR: _handle_xor_node,
    _EQUIVALENT_TO: _handle_equivalence_node,
}


def _handle_call(visitor, node, *args):
    return _call_dispatchers[args[0][0]](visitor, node, args[0][1], *args[1:])


##
# Scheduling
##

# This will get populated when cp is finally imported
_before_dispatchers = {}
_at_dispatchers = {}
_time_point_dispatchers = {_GENERAL: lambda x: x, _ELEMENT_CONSTRAINT: lambda x: x}

_non_precedence_types = {_GENERAL, _ELEMENT_CONSTRAINT}


def _handle_before_expression_node(visitor, node, time1, time2, delay):
    t1 = (_GENERAL, _time_point_dispatchers[time1[0]](time1[1]))
    t2 = (_GENERAL, _time_point_dispatchers[time2[0]](time2[1]))
    lhs = _handle_sum_node(visitor, None, t1, delay)
    if time1[0] in _non_precedence_types or time2[0] in _non_precedence_types:
        # we already know we can't use a start_before_start function or its ilk:
        # Just build the correct inequality.
        return _handle_inequality_node(visitor, None, lhs, t2)

    # If this turns out to be the root, we can use the second return, but we
    # also pass the args for the inequality expression in case we use this in a
    # boolean-valued context.
    return (
        _BEFORE,
        _before_dispatchers[time1[0], time2[0]](time1[1], time2[1], delay[1]),
        (lhs, t2),
    )


def _handle_at_expression_node(visitor, node, time1, time2, delay):
    t1 = (_GENERAL, _time_point_dispatchers[time1[0]](time1[1]))
    t2 = (_GENERAL, _time_point_dispatchers[time2[0]](time2[1]))
    lhs = _handle_sum_node(visitor, None, t1, delay)
    if time1[0] in _non_precedence_types or time2[0] in _non_precedence_types:
        # we can't use a start_before_start function or its ilk: Just build the
        # correct inequality.
        return _handle_equality_node(visitor, None, lhs, t2)

    return (
        _AT,
        _at_dispatchers[time1[0], time2[0]](time1[1], time2[1], delay[1]),
        (lhs, t2),
    )


def _handle_always_in_node(visitor, node, cumul_func, lb, ub, start, end):
    return (
        _GENERAL,
        cp.always_in(cumul_func[1], interval=(start[1], end[1]), min=lb[1], max=ub[1]),
    )


class LogicalToDoCplex(StreamBasedExpressionVisitor):
    _operator_handles = {
        EXPR.GetItemExpression: _handle_getitem,
        EXPR.Structural_GetItemExpression: _handle_getitem,
        EXPR.Numeric_GetItemExpression: _handle_getitem,
        EXPR.Boolean_GetItemExpression: _handle_getitem,
        EXPR.GetAttrExpression: _handle_getattr,
        EXPR.Structural_GetAttrExpression: _handle_getattr,
        EXPR.Numeric_GetAttrExpression: _handle_getattr,
        EXPR.Boolean_GetAttrExpression: _handle_getattr,
        EXPR.CallExpression: _handle_call,
        EXPR.NegationExpression: _handle_negation_node,
        EXPR.ProductExpression: _handle_product_node,
        EXPR.DivisionExpression: _handle_division_node,
        EXPR.PowExpression: _handle_pow_node,
        EXPR.AbsExpression: _handle_abs_node,
        EXPR.MonomialTermExpression: _handle_monomial_expr,
        EXPR.SumExpression: _handle_sum_node,
        EXPR.LinearExpression: _handle_sum_node,
        EXPR.MinExpression: _handle_min_node,
        EXPR.MaxExpression: _handle_max_node,
        EXPR.NotExpression: _handle_not_node,
        EXPR.EquivalenceExpression: _handle_equivalence_node,
        EXPR.ImplicationExpression: _handle_implication_node,
        EXPR.AndExpression: _handle_and_node,
        EXPR.OrExpression: _handle_or_node,
        EXPR.XorExpression: _handle_xor_node,
        EXPR.ExactlyExpression: _handle_exactly_node,
        EXPR.AtMostExpression: _handle_at_most_node,
        EXPR.AtLeastExpression: _handle_at_least_node,
        EXPR.EqualityExpression: _handle_equality_node,
        EXPR.NotEqualExpression: _handle_not_equal_node,
        EXPR.InequalityExpression: _handle_inequality_node,
        EXPR.RangedExpression: _handle_ranged_inequality_node,
        BeforeExpression: _handle_before_expression_node,
        AtExpression: _handle_at_expression_node,
        AlwaysIn: _handle_always_in_node,
        _GeneralExpressionData: _handle_named_expression_node,
        ScalarExpression: _handle_named_expression_node,
    }
    _var_handles = {
        IntervalVarStartTime: _before_interval_var_start_time,
        IntervalVarEndTime: _before_interval_var_end_time,
        IntervalVarLength: _before_interval_var_length,
        IntervalVarPresence: _before_interval_var_presence,
        ScalarIntervalVar: _before_interval_var,
        IntervalVarData: _before_interval_var,
        IndexedIntervalVar: _before_indexed_interval_var,
        ScalarVar: _before_var,
        _GeneralVarData: _before_var,
        IndexedVar: _before_indexed_var,
        ScalarBooleanVar: _before_boolean_var,
        _GeneralBooleanVarData: _before_boolean_var,
        IndexedBooleanVar: _before_indexed_boolean_var,
        _GeneralExpressionData: _before_named_expression,
        ScalarExpression: _before_named_expression,
        IndexedParam: _before_indexed_param,  # Because of indirection
        ScalarParam: _before_param,
    }

    def __init__(self, cpx_model, symbolic_solver_labels=False):
        self.cpx = cpx_model
        self.symbolic_solver_labels = symbolic_solver_labels
        self._process_node = self._process_node_bx

        self.var_map = {}
        self._named_expressions = {}
        self.pyomo_to_docplex = ComponentMap()

    def initializeWalker(self, expr):
        expr, src, src_idx = expr
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            return False, result
        return True, expr

    def beforeChild(self, node, child, child_idx):
        # Return native types
        if child.__class__ in EXPR.native_types:
            return False, (_GENERAL, child)

        if child.__class__ in step_func_expression_types:
            return _step_function_handles[child.__class__](self, child)

        # Convert Vars Logical vars to docplex equivalents
        if not child.is_expression_type() or child.is_named_expression_type():
            return self._var_handles[child.__class__](self, child)

        return True, None

    def exitNode(self, node, data):
        return self._operator_handles[node.__class__](self, node, *data)

    finalizeResult = None


# [ESJ 11/7/22]: TODO: We should revisit this method in the future, as it is not
# very efficient.
def collect_valid_components(model, active=True, sort=None, valid=set(), targets=set()):
    assert active in (True, None)
    unrecognized = {}
    components = {k: [] for k in targets}
    for obj in model.component_data_objects(active=True, descend_into=True, sort=sort):
        ctype = obj.ctype
        if ctype in components:
            components[ctype].append(obj)
        elif ctype not in valid:
            if ctype not in unrecognized:
                unrecognized[ctype] = [obj]
            else:
                unrecognized[ctype].append(obj)

    return components, unrecognized


@WriterFactory.register(
    'docplex_model', 'Generate the corresponding docplex model object'
)
class DocplexWriter(object):
    CONFIG = ConfigDict('docplex_model_writer')
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write Pyomo Var and Constraint names to docplex model',
        ),
    )

    def __init__(self):
        self.config = self.CONFIG()

    def write(self, model, **options):
        config = options.pop('config', self.config)(options)

        components, unknown = collect_valid_components(
            model,
            active=True,
            sort=SortComponents.deterministic,
            valid={
                Block,
                Objective,
                Constraint,
                Var,
                Param,
                BooleanVar,
                LogicalConstraint,
                Suffix,
                # FIXME: Non-active components should not report as Active
                Set,
                RangeSet,
                Port,
            },
            targets={Objective, Constraint, LogicalConstraint, IntervalVar},
        )
        if unknown:
            raise ValueError(
                "The model ('%s') contains the following active components "
                "that the docplex writer does not know how to process:\n\t%s"
                % (
                    model.name,
                    "\n\t".join(
                        "%s:\n\t\t%s" % (k, "\n\t\t".join(map(attrgetter('name'), v)))
                        for k, v in unknown.items()
                    ),
                )
            )

        cpx_model = cp.CpoModel()
        visitor = LogicalToDoCplex(
            cpx_model, symbolic_solver_labels=config.symbolic_solver_labels
        )

        active_objs = components[Objective]
        # [ESJ 09/29/22]: TODO: I think that CP Optimizer can support
        # multiple objectives. We should generalize this later, but for
        # now I don't much care.
        if len(active_objs) > 1:
            raise ValueError(
                "More than one active objective defined for "
                "input model '%s': Cannot write to docplex." % model.name
            )
        elif len(active_objs) == 1:
            obj = active_objs[0]
            obj_expr = visitor.walk_expression((obj.expr, obj, 0))
            if obj.sense is minimize:
                cpx_model.add(cp.minimize(obj_expr[1]))
            else:
                cpx_model.add(cp.maximize(obj_expr[1]))

        # No objective is fine too, this is CP afterall...

        # Write algebraic constraints
        for cons in components[Constraint]:
            expr = visitor.walk_expression((cons.body, cons, 0))
            if cons.lower is not None and cons.upper is not None:
                cpx_model.add(cp.range(expr[1], lb=cons.lb, ub=cons.ub))
            elif cons.lower is not None:
                cpx_model.add(cons.lb <= expr[1])
            elif cons.upper is not None:
                cpx_model.add(cons.ub >= expr[1])

        # Write interval vars (these are secretly constraints if they have to be
        # scheduled)
        for var in components[IntervalVar]:
            # we just walk it so it gets added to the model. Note that
            # adding it again here would add it for a second time, so that's
            # why we don't.
            visitor.walk_expression((var, var, 0))

        # Write logical constraints
        for cons in components[LogicalConstraint]:
            expr = visitor.walk_expression((cons.expr, cons, 0))
            if expr[0] is _ELEMENT_CONSTRAINT:
                # Make the expression into a docplex-approved boolean-valued
                # expression, if it turned out that the root of the
                # expression was just an element constraint. (This can
                # happen for something like a constraint that requires that
                # an interval var specified by indirection has to be
                # present.)
                cpx_model.add(expr[1] == True)
            else:
                cpx_model.add(expr[1])

        # That's all, folks.
        return cpx_model, visitor.pyomo_to_docplex


@SolverFactory.register('cp_optimizer', doc='Direct interface to CPLEX CP Optimizer')
class CPOptimizerSolver(object):
    CONFIG = ConfigDict("cp_optimizer_solver")
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description='Write Pyomo Var and Constraint names to docplex model',
        ),
    )
    CONFIG.declare(
        'tee',
        ConfigValue(
            default=False, domain=bool, description="Stream solver output to terminal."
        ),
    )
    CONFIG.declare(
        'options', ConfigValue(default={}, description="Dictionary of solver options.")
    )

    _unrestricted_license = None

    def __init__(self, **kwds):
        self.config = self.CONFIG()
        self.config.set_value(kwds)
        if docplex_available:
            self._solve_status_map = {
                cp.SOLVE_STATUS_UNKNOWN: TerminationCondition.unknown,
                cp.SOLVE_STATUS_INFEASIBLE: TerminationCondition.infeasible,
                cp.SOLVE_STATUS_FEASIBLE: TerminationCondition.feasible,
                cp.SOLVE_STATUS_OPTIMAL: TerminationCondition.optimal,
                cp.SOLVE_STATUS_JOB_ABORTED: None,  # we need the fail status
                cp.SOLVE_STATUS_JOB_FAILED: TerminationCondition.solverFailure,
            }
            self._stop_cause_map = {
                # We only need to check this if we get an 'aborted' status, so
                # if this says it hasn't been stopped, we're just confused at
                # this point.
                cp.STOP_CAUSE_NOT_STOPPED: TerminationCondition.unknown,
                cp.STOP_CAUSE_LIMIT: TerminationCondition.maxTimeLimit,
                # User called exit, maybe in a callback.
                cp.STOP_CAUSE_EXIT: TerminationCondition.userInterrupt,
                # docplex says "Search aborted externally"
                cp.STOP_CAUSE_ABORT: TerminationCondition.userInterrupt,
                # This is in their documentation, but not here, for some reason
                # cp.STOP_CAUSE_UNKNOWN: TerminationCondition.unknown
            }

    @property
    def options(self):
        return self.config.options

    # Support use as a context manager under current solver API
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        return Executable('cpoptimizer').available() and docplex_available

    def license_is_valid(self):
        if CPOptimizerSolver._unrestricted_license is None:
            # Note: 140*log_2(140) == 998.1 fits in CE,
            #     141*log_2(141) == 1006.7 does not
            x = cp.integer_var_list(141, 1, 141, "X")
            m = cp.CpoModel()
            m.add(cp.all_diff(x))
            try:
                m.solve()
                CPOptimizerSolver._unrestricted_license = True
            except cp_solver.solver.CpoSolverException:
                CPOptimizerSolver._unrestricted_license = False
        return CPOptimizerSolver._unrestricted_license

    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        config = self.config()
        config.set_value(kwds)

        writer = DocplexWriter()
        cpx_model, var_map = writer.write(
            model, symbolic_solver_labels=config.symbolic_solver_labels
        )
        if not config.tee:
            # If the user has also set LogVerbosity, we'll assume they know what
            # they're doing.
            verbosity = config.options.get('LogVerbosity')
            if verbosity is None:
                config.options['LogVerbosity'] = 'Quiet'

        msol = cpx_model.solve(**self.options)

        # Transfer the solver status to the pyomo results object
        results = SolverResults()
        results.solver.name = "CP Optimizer"
        results.problem.name = model.name

        info = msol.get_solver_infos()
        results.problem.number_of_constraints = info.get_number_of_constraints()
        int_vars = info.get_number_of_integer_vars()
        interval_vars = info.get_number_of_interval_vars()
        results.problem.number_of_integer_vars = int_vars
        results.problem.number_of_interval_vars = interval_vars
        # This is a useless number, but so is 0, so...
        results.problem.number_of_variables = int_vars + interval_vars

        val = msol.get_objective_value()
        bound = msol.get_objective_bound()
        if cpx_model.is_maximization():
            results.problem.number_of_objectives = 1
            results.problem.sense = maximize
            results.problem.lower_bound = val
            results.problem.upper_bound = bound
        elif cpx_model.is_minimization():
            results.problem.number_of_objectives = 1
            results.problem.sense = minimize
            results.problem.lower_bound = bound
            results.problem.upper_bound = val
        else:
            # it's a satisfaction problem
            results.problem.number_of_objectives = 0
            results.problem.sense = None
            results.problem.lower_bound = None
            results.problem.upper_bound = None

        results.solver.solve_time = msol.get_solve_time()
        solve_status = msol.get_solve_status()
        results.solver.termination_condition = (
            self._solve_status_map[solve_status]
            if solve_status is not None
            else self._stop_cause_map[msol.get_stop_cause()]
        )

        # Copy the variable values onto the Pyomo model, using the map we stored
        # on the writer.
        cp_sol = msol.get_solution()
        if cp_sol is not None:
            for py_var, cp_var in var_map.items():
                sol = cp_sol.get_var_solution(cp_var)
                if sol is None:
                    logger.warning(
                        "CP optimizer did not return a value "
                        "for variable '%s'" % py_var.name
                    )
                else:
                    sol = sol.get_value()
                if py_var.ctype is IntervalVar:
                    if len(sol) == 0:
                        # The interval_var is absent
                        py_var.is_present.set_value(False)
                    else:
                        (start, end, size) = sol
                        py_var.is_present.set_value(True)
                        py_var.start_time.set_value(start, skip_validation=True)
                        py_var.end_time.set_value(end, skip_validation=True)
                        py_var.length.set_value(end - start, skip_validation=True)
                elif py_var.ctype in {Var, BooleanVar}:
                    py_var.set_value(sol, skip_validation=True)
                else:
                    raise DeveloperError(
                        "Unrecognized Pyomo type in pyomo-to-docplex "
                        "variable map: %s" % type(py_var)
                    )

        return results
