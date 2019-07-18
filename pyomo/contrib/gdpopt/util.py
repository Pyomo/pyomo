"""Utility functions and classes for the GDPopt solver."""
from __future__ import division

import logging
import six
from math import fabs, floor, log

from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (Block, Constraint,
                        Objective, Reals, Var, minimize, value, Expression)
from pyomo.core.expr.current import identify_variables
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory
from pyomo.opt.results import ProblemSense
from pyomo.common.log import LoggingIntercept
import timeit
from contextlib import contextmanager
from pyomo.util.model_size import build_model_size_report


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


class SuppressInfeasibleWarning(LoggingIntercept):
    """Suppress the infeasible model warning message from solve().

    The "WARNING: Loading a SolverResults object with a warning status" warning
    message from calling solve() is often unwanted, but there is no clear way
    to suppress it.

    """

    def __init__(self):
        super(SuppressInfeasibleWarning, self).__init__(
            six.StringIO(), 'pyomo.core', logging.WARNING)


def model_is_valid(solve_data, config):
    """Validate that the model is solvable by GDPopt.

    """
    m = solve_data.working_model
    GDPopt = m.GDPopt_utils

    # Handle LP/NLP being passed to the solver
    prob = solve_data.results.problem
    if (prob.number_of_binary_variables == 0 and
            prob.number_of_integer_variables == 0 and
            prob.number_of_disjunctions == 0):
        config.logger.info('Problem has no discrete decisions.')
        obj = next(m.component_data_objects(Objective, active=True))
        if (any(c.body.polynomial_degree() not in (1, 0) for c in GDPopt.constraint_list) or
                obj.expr.polynomial_degree() not in (1, 0)):
            config.logger.info(
                "Your model is an NLP (nonlinear program). "
                "Using NLP solver %s to solve." % config.nlp_solver)
            SolverFactory(config.nlp_solver).solve(
                solve_data.original_model, **config.nlp_solver_args)
            return False
        else:
            config.logger.info(
                "Your model is an LP (linear program). "
                "Using LP solver %s to solve." % config.mip_solver)
            SolverFactory(config.mip_solver).solve(
                solve_data.original_model, **config.mip_solver_args)
            return False

    # TODO if any continuous variables are multipled with binary ones, need
    # to do some kind of transformation (Glover?) or throw an error message
    return True


def process_objective(solve_data, config, move_linear_objective=False):
    m = solve_data.working_model
    util_blk = getattr(m, solve_data.util_block_name)
    # Handle missing or multiple objectives
    active_objectives = list(m.component_data_objects(
        ctype=Objective, active=True, descend_into=True))
    solve_data.results.problem.number_of_objectives = len(active_objectives)
    if len(active_objectives) == 0:
        config.logger.warning(
            'Model has no active objectives. Adding dummy objective.')
        util_blk.dummy_objective = Objective(expr=1)
        main_obj = util_blk.dummy_objective
    elif len(active_objectives) > 1:
        raise ValueError('Model has multiple active objectives.')
    else:
        main_obj = active_objectives[0]
    solve_data.results.problem.sense = main_obj.sense
    solve_data.objective_sense = main_obj.sense

    # Move the objective to the constraints if it is nonlinear
    if main_obj.expr.polynomial_degree() not in (1, 0) \
            or move_linear_objective:
        if move_linear_objective:
            config.logger.info("Moving objective to constraint set.")
        else:
            config.logger.info("Objective is nonlinear. Moving it to constraint set.")

        util_blk.objective_value = Var(domain=Reals, initialize=0)
        if mcpp_available():
            mc_obj = McCormick(main_obj.expr)
            util_blk.objective_value.setub(mc_obj.upper())
            util_blk.objective_value.setlb(mc_obj.lower())

        if main_obj.sense == minimize:
            util_blk.objective_constr = Constraint(
                expr=util_blk.objective_value >= main_obj.expr)
            solve_data.results.problem.sense = ProblemSense.minimize
        else:
            util_blk.objective_constr = Constraint(
                expr=util_blk.objective_value <= main_obj.expr)
            solve_data.results.problem.sense = ProblemSense.maximize
        # Deactivate the original objective and add this new one.
        main_obj.deactivate()
        util_blk.objective = Objective(
            expr=util_blk.objective_value, sense=main_obj.sense)
        # Add the new variable and constraint to the working lists
        util_blk.variable_list.append(util_blk.objective_value)
        util_blk.constraint_list.append(util_blk.objective_constr)


def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)


def copy_var_list_values(from_list, to_list, config,
                         skip_stale=False, skip_fixed=True,
                         ignore_integrality=False):
    """Copy variable values from one list to another.

    Rounds to Binary/Integer if neccessary
    Sets to zero for NonNegativeReals if neccessary
    """
    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue  # Skip stale variable values.
        if skip_fixed and v_to.is_fixed():
            continue  # Skip fixed variables.
        try:
            v_to.set_value(value(v_from, exception=False))
            if skip_stale:
                v_to.stale = False
        except ValueError as err:
            err_msg = getattr(err, 'message', str(err))
            var_val = value(v_from)
            rounded_val = int(round(var_val))
            # Check to see if this is just a tolerance issue
            if ignore_integrality \
                and ('is not in domain Binary' in err_msg
                or 'is not in domain Integers' in err_msg):
               v_to.value = value(v_from, exception=False)
            elif 'is not in domain Binary' in err_msg and (
                    fabs(var_val - 1) <= config.integer_tolerance or
                    fabs(var_val) <= config.integer_tolerance):
                v_to.set_value(rounded_val)
            # TODO What about PositiveIntegers etc?
            elif 'is not in domain Integers' in err_msg and (
                    fabs(var_val - rounded_val) <= config.integer_tolerance):
                v_to.set_value(rounded_val)
            # Value is zero, but shows up as slightly less than zero.
            elif 'is not in domain NonNegativeReals' in err_msg and (
                    fabs(var_val) <= config.zero_tolerance):
                v_to.set_value(0)
            else:
                raise


def is_feasible(model, config):
    """Checks to see if the algebraic model is feasible in its current state.

    Checks variable bounds and active constraints. Not for use with
    untransformed GDP models.

    """
    disj = next(model.component_data_objects(
        ctype=Disjunct, active=True), None)
    if disj is not None:
        raise NotImplementedError(
            "Found active disjunct %s. "
            "This function is not intended to check "
            "feasibility of disjunctive models, "
            "only transformed subproblems." % disj.name)

    config.logger.debug('Checking if model is feasible.')
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
            config.logger.info('%s: value %s < LB %s' % (
                var.name, value(var), value(var.lb)))
            return False
        # Check variable upper bound
        if (var.has_ub() and
                value(var) - value(var.ub) >= config.variable_tolerance):
            config.logger.info('%s: value %s > UB %s' % (
                var.name, value(var), value(var.ub)))
            return False
    config.logger.info('Model is feasible.')
    return True


def build_ordered_component_lists(model, solve_data):
    """Define lists used for future data transfer.

    Also attaches ordered lists of the variables, constraints, disjuncts, and
    disjunctions to the model so that they can be used for mapping back and
    forth.

    """
    util_blk = getattr(model, solve_data.util_block_name)
    var_set = ComponentSet()
    setattr(
        util_blk, 'constraint_list', list(
            model.component_data_objects(
                ctype=Constraint, active=True,
                descend_into=(Block, Disjunct))))
    setattr(
        util_blk, 'disjunct_list', list(
            model.component_data_objects(
                ctype=Disjunct, descend_into=(Block, Disjunct))))
    setattr(
        util_blk, 'disjunction_list', list(
            model.component_data_objects(
                ctype=Disjunction, active=True,
                descend_into=(Disjunct, Block))))

    # Identify the non-fixed variables in (potentially) active constraints and
    # objective functions
    for constr in getattr(util_blk, 'constraint_list'):
        for v in identify_variables(constr.body, include_fixed=False):
            var_set.add(v)
    for obj in model.component_data_objects(ctype=Objective, active=True):
        for v in identify_variables(obj.expr, include_fixed=False):
            var_set.add(v)
    # Disjunct indicator variables might not appear in active constraints. In
    # fact, if we consider them Logical variables, they should not appear in
    # active algebraic constraints. For now, they need to be added to the
    # variable set.
    for disj in getattr(util_blk, 'disjunct_list'):
        var_set.add(disj.indicator_var)

    # We use component_data_objects rather than list(var_set) in order to
    # preserve a deterministic ordering.
    var_list = list(
        v for v in model.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
        if v in var_set)
    setattr(util_blk, 'variable_list', var_list)


def setup_results_object(solve_data, config):
    """Record problem statistics for original model."""
    # Create the solver results object
    res = solve_data.results
    prob = res.problem
    res.problem.name = solve_data.original_model.name
    res.problem.number_of_nonzeros = None  # TODO
    # TODO work on termination condition and message
    res.solver.termination_condition = None
    res.solver.message = None
    res.solver.user_time = None
    res.solver.system_time = None
    res.solver.wallclock_time = None
    res.solver.termination_message = None

    num_of = build_model_size_report(solve_data.original_model)

    # Get count of constraints and variables
    prob.number_of_constraints = num_of.activated.constraints
    prob.number_of_disjunctions = num_of.activated.disjunctions
    prob.number_of_variables = num_of.activated.variables
    prob.number_of_binary_variables = num_of.activated.binary_variables
    prob.number_of_continuous_variables = num_of.activated.continuous_variables
    prob.number_of_integer_variables = num_of.activated.integer_variables

    config.logger.info(
        "Original model has %s constraints (%s nonlinear) "
        "and %s disjunctions, "
        "with %s variables, of which %s are binary, %s are integer, "
        "and %s are continuous." %
        (num_of.activated.constraints,
         num_of.activated.nonlinear_constraints,
         num_of.activated.disjunctions,
         num_of.activated.variables,
         num_of.activated.binary_variables,
         num_of.activated.integer_variables,
         num_of.activated.continuous_variables))


# def validate_disjunctions(model, config):
#     """Validate that the active disjunctions on the model are satisfied
#     by the current disjunct indicator_var values."""
#     active_disjunctions = model.component_data_objects(
#         ctype=Disjunction, active=True, descend_into=(Block, Disjunct))
#     for disjtn in active_disjunctions:
#         sum_disj_vals = sum(disj.indicator_var.value
#                             for disj in disjtn.disjuncts)
#         if disjtn.xor and fabs(sum_disj_vals - 1) > config.integer_tolerance:
#             raise ValueError(
#                 "Expected disjunct values to add up to 1 "
#                 "for XOR disjunction %s. "
#                 "Instead, values add up to %s." % (disjtn.name, sum_disj_vals))
#         elif sum_disj_vals + config.integer_tolerance < 1:
#             raise ValueError(
#                 "Expected disjunct values to add up to at least 1 for "
#                 "OR disjunction %s. "
#                 "Instead, values add up to %s." % (disjtn.name, sum_disj_vals))


def constraints_in_True_disjuncts(model, config):
    """Yield constraints in disjuncts where the indicator value is set or fixed to True."""
    for constr in model.component_data_objects(Constraint):
        yield constr
    observed_disjuncts = ComponentSet()
    for disjctn in model.component_data_objects(Disjunction):
        # get all the disjuncts in the disjunction. Check which ones are True.
        for disj in disjctn.disjuncts:
            if disj in observed_disjuncts:
                continue
            observed_disjuncts.add(disj)
            if fabs(disj.indicator_var.value - 1) <= config.integer_tolerance:
                for constr in disj.component_data_objects(Constraint):
                    yield constr


@contextmanager
def time_code(timing_data_obj, code_block_name, is_main_timer=False):
    """Starts timer at entry, stores elapsed time at exit

    If `is_main_timer=True`, the start time is stored in the timing_data_obj,
    allowing calculation of total elapsed time 'on the fly' (e.g. to enforce
    a time limit) using `get_main_elapsed_time(timing_data_obj)`.
    """
    start_time = timeit.default_timer()
    if is_main_timer:
        timing_data_obj.main_timer_start_time = start_time
    yield
    elapsed_time = timeit.default_timer() - start_time
    prev_time = timing_data_obj.get(code_block_name, 0)
    timing_data_obj[code_block_name] = prev_time + elapsed_time


def get_main_elapsed_time(timing_data_obj):
    """Returns the time since entering the main `time_code` context"""
    current_time = timeit.default_timer()
    try:
        return current_time - timing_data_obj.main_timer_start_time
    except AttributeError as e:
        if 'main_timer_start_time' in str(e):
            six.raise_from(e, AttributeError(
                "You need to be in a 'time_code' context to use `get_main_elapsed_time()`."
            ))


@contextmanager
def restore_logger_level(logger):
    old_logger_level = logger.getEffectiveLevel()
    yield
    logger.setLevel(old_logger_level)


@contextmanager
def create_utility_block(model, name, solve_data):
    created_util_block = False
    # Create a model block on which to store GDPopt-specific utility
    # modeling objects.
    if hasattr(model, name):
        raise RuntimeError(
            "GDPopt needs to create a Block named %s "
            "on the model object, but an attribute with that name "
            "already exists." % name)
    else:
        created_util_block = True
        setattr(model, name, Block(
            doc="Container for GDPopt solver utility modeling objects"))
        solve_data.util_block_name = name

        # Save ordered lists of main modeling components, so that data can
        # be easily transferred between future model clones.
        build_ordered_component_lists(model, solve_data)
    yield
    if created_util_block:
        model.del_component(name)
