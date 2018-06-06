"""Utility functions and classes for the GDPopt solver."""
from __future__ import division

import logging
from math import fabs, floor, log

from pyomo.core import (Any, Binary, Block, Constraint, NonNegativeReals, Var,
                        value)
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel import ComponentSet
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.results import SolverResults


class _DoNothing(object):
    """Do nothing, literally.

    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing


class GDPoptSolveData(object):
    """Data container to hold solve-instance data."""
    pass


def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)


def copy_var_list_values(from_list, to_list, config, skip_stale=False):
    """Copy variable values from one list to another."""
    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue  # Skip stale variable values.
        try:
            v_to.set_value(value(v_from, exception=False))
            if skip_stale:
                v_to.stale = False
        except ValueError as err:
            if 'is not in domain Binary' in err.message:
                # Check to see if this is just a tolerance issue
                v_from_val = value(v_from, exception=False)
                if (fabs(v_from_val - 1) <= config.integer_tolerance or
                        fabs(v_from_val) <= config.integer_tolerance):
                    v_to.set_value(round(v_from_val))
                else:
                    raise


def is_feasible(model, config):
    config.logger.info('Checking if model is feasible.')
    for constr in model.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        # Check constraint lower bound
        if (constr.lower is not None and (
            value(constr.lower) - value(constr.body)
            >= config.constraint_tolerance
        )):
            config.logger.info('%s: body %s < LB %s' % (
                constr.name, value(constr.body), value(constr.lower)))
            return False
        # check constraint upper bound
        if (constr.upper is not None and (
            value(constr.body) - value(constr.upper)
            >= config.constraint_tolerance
        )):
            config.logger.info('%s: body %s > UB %s' % (
                constr.name, value(constr.body), value(constr.upper)))
            return False
    for var in model.component_data_objects(ctype=Var, descend_into=True):
        # Check variable lower bound
        if (var.has_lb() and
                value(var.lb) - value(var) >= config.variable_tolerance):
            config.logger.info('%s: %s < LB %s' % (
                var.name, value(var), value(var.lb)))
            return False
        # Check variable upper bound
        if (var.has_ub() and
                value(var) - value(var.ub) >= config.variable_tolerance):
            config.logger.info('%s: %s > UB %s' % (
                var.name, value(var), value(var.ub)))
            return False
    config.logger.info('Model is feasible.')
    return True


def _define_initial_ordered_component_lists(model):
    """Define lists used for future data transfer."""
    GDPopt = model.GDPopt_utils
    var_set = ComponentSet()
    GDPopt.initial_constraints_list = list(
        model.component_data_objects(
            ctype=Constraint, active=True, descend_into=(Block, Disjunct)))
    GDPopt.initial_disjuncts_list = list(
        model.component_data_objects(
            ctype=Disjunct, descend_into=(Block, Disjunct)))
    GDPopt.initial_disjunctions_list = list(
        model.component_data_objects(
            ctype=Disjunction, active=True,
            descend_into=(Disjunct, Block)))

    # Identify the non-fixed variables in (potentially) active constraints
    for constr in GDPopt.initial_constraints_list:
        for v in EXPR.identify_variables(constr.body, include_fixed=False):
            var_set.add(v)
    # Disjunct indicator variables might not appear in active constraints. In
    # fact, if we consider them Logical variables, they should not appear in
    # active algebraic constraints. For now, they need to be added to the
    # variable set.
    for disj in GDPopt.initial_disjuncts_list:
        var_set.add(disj.indicator_var)

    # We use component_data_objects rather than list(var_set) in order to
    # preserve a deterministic ordering.
    GDPopt.initial_var_list = list(
        v for v in model.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
        if v in var_set)
    GDPopt.initial_nonlinear_constraints = [
        v for v in GDPopt.initial_constraints_list
        if v.body.polynomial_degree() not in (0, 1)]


def _record_problem_statistics(model, solve_data, config):
    # Create the solver results object
    res = solve_data.results = SolverResults()
    prob = res.problem
    GDPopt = model.GDPopt_utils
    res.problem.name = model.name
    res.problem.number_of_nonzeros = None  # TODO
    # TODO work on termination condition and message
    res.solver.termination_condition = None
    res.solver.message = None
    # TODO add some kind of timing
    res.solver.user_time = None
    res.solver.system_time = None
    res.solver.wallclock_time = None
    res.solver.termination_message = None

    # Classify the variables
    num_binary, num_continuous = 0, 0
    for v in GDPopt.initial_var_list:
        if v.is_binary():
            num_binary += 1
        elif v.is_continuous():
            num_continuous += 1
        elif v.is_integer():
            raise TypeError(
                "GDP model has unreformulated integer variable %s"
                % v.name)
        else:
            raise TypeError('Variable {0} has unknown domain of {1}'.
                            format(v.name, v.domain))

    # Get count of constraints and variables
    prob.number_of_constraints = len(GDPopt.initial_constraints_list)
    prob.number_of_disjunctions = len(GDPopt.initial_disjunctions_list)

    prob.number_of_variables = len(GDPopt.initial_var_list)
    prob.number_of_binary_variables = num_binary
    prob.number_of_continuous_variables = num_continuous
    prob.number_of_integer_variables = 0
    config.logger.info(
        "Problem has %s constraints (%s nonlinear) and %s disjunctions, "
        "with %s variables, of which %s are binary and %s continuous." %
        (prob.number_of_constraints,
         len(GDPopt.initial_nonlinear_constraints),
         prob.number_of_disjunctions,
         prob.number_of_variables,
         prob.number_of_binary_variables,
         prob.number_of_continuous_variables))


def reformulate_integer_variables(model, config):
    integer_vars = list(
        v for v in model.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
        if v.is_integer() and not v.fixed)
    if len(integer_vars) == 0:
        return  # if no free integer variables, no reformulation needed.

    if config.reformulate_integer_vars_using is None:
        config.logger.warning(
            "Model contains unfixed integer variables. "
            "GDPopt will reformulate using base 2 binary variables "
            "by default. To specify a different method, see the "
            "reformulate_integer_vars_using configuration option.")
        config.reformulate_integer_vars_using = 'base2_binary'

    config.logger.info(
        "Reformulating integer variables using the %s strategy."
        % config.reformulate_integer_vars_using)

    # Set up reformulation block
    reform_block = model.GDPopt_utils.integer_reform = Block(
        doc="Holds variables and constraints for reformulating "
        "integer variables to binary variables.")
    reform_block.new_binary_var = Var(
        Any, domain=Binary, dense=False,
        doc="Binary variable with index (int_var.name, indx)")
    reform_block.integer_to_binary_constraint = Constraint(
        Any, doc="Equality constraints mapping the binary variable values "
        "to the integer variable value.")

    # check that variables are bounded and non-negative
    for int_var in integer_vars:
        if not (int_var.has_lb() and int_var.has_ub()):
            raise ValueError(
                "Integer variable %s is missing an "
                "upper or lower bound. LB: %s; UB: %s. "
                "GDPopt does not support unbounded integer variables."
                % (int_var.name, int_var.lb, int_var.ub))
        if int_var.lb < 0:
            raise ValueError(
                "Integer variable %s can be negative. "
                "GDPopt currently only supports positive integer "
                "variables." % (int_var.name)
            )
        # do the reformulation
        highest_power = floor(log(value(int_var.ub), 2))
        var_name = int_var.name
        reform_block.integer_to_binary_constraint.add(
            var_name, expr=int_var == sum(
                reform_block.new_binary_var[var_name, pwr] * (2 ^ pwr)
                for pwr in range(0, int(highest_power) + 1)))
        int_var.domain = NonNegativeReals

    config.logger.info(
        "Reformulated %s integer variables using "
        "%s binary variables and %s constraints."
        % (len(integer_vars), len(reform_block.new_binary_var),
           len(reform_block.integer_to_binary_constraint)))
