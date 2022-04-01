#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Utility functions and classes for the GDPopt solver."""

from contextlib import contextmanager
import logging
from math import fabs
import sys

from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
    Block, Constraint, Objective, Reals, Var, minimize, value)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory

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

# ESJ TODO: I think we also need to suppress the one about not being able to
# load results into the model...
class SuppressInfeasibleWarning(object):
    """Suppress the infeasible model warning message from solve().

    The "WARNING: Loading a SolverResults object with a warning status" warning
    message from calling solve() is often unwanted, but there is no clear way
    to suppress it.

    This is modeled on LoggingIntercept from pyomo.common.log,
    but different in function.

    """

    class InfeasibleWarningFilter(logging.Filter):
        def filter(self, record):
            return not record.getMessage().startswith(
                "Loading a SolverResults object with a warning status into "
                "model=")

    warning_filter = InfeasibleWarningFilter()

    def __enter__(self):
        logger = logging.getLogger('pyomo.core')
        logger.addFilter(self.warning_filter)

    def __exit__(self, exception_type, exception_value, traceback):
        logger = logging.getLogger('pyomo.core')
        logger.removeFilter(self.warning_filter)

def solve_continuous_problem(m, config):
    logger = config.logger
    logger.info('Problem has no discrete decisions.')
    obj = next(m.component_data_objects(Objective, active=True))
    if (any(c.body.polynomial_degree() not in (1, 0) for c in
            m.component_data_objects(Constraint, active=True,
                                     descend_into=Block)) 
        or obj.expr.polynomial_degree() not in (1, 0)):
        logger.info("Your model is an NLP (nonlinear program). "
                    "Using NLP solver %s to solve." % config.nlp_solver)
        results = SolverFactory(config.nlp_solver).solve( 
            m, **config.nlp_solver_args)
        return results
    else:
        logger.info("Your model is an LP (linear program). "
                    "Using LP solver %s to solve." % config.mip_solver)
        results = SolverFactory(config.mip_solver).solve(
            m, **config.mip_solver_args)
        return results

def move_nonlinear_objective_to_constraints(util_block, logger):
    m = util_block.model()
    main_obj = next(m.component_data_objects(Objective, descend_into=True,
                                             active=True))

    # Move the objective to the constraints if it is nonlinear
    if main_obj.expr.polynomial_degree() not in (1, 0):
        logger.info("Objective is nonlinear. Moving it to constraint set.")

        util_block.objective_value = Var(domain=Reals, initialize=0)
        if mcpp_available():
            mc_obj = McCormick(main_obj.expr)
            util_block.objective_value.setub(mc_obj.upper())
            util_block.objective_value.setlb(mc_obj.lower())
        else:
            # Use Pyomo's contrib.fbbt package
            lb, ub = compute_bounds_on_expr(main_obj.expr)
            if main_obj.sense == minimize:
                util_block.objective_value.setlb(lb)
            else:
                util_block.objective_value.setub(ub)

        if main_obj.sense == minimize:
            util_block.objective_constr = Constraint(
                expr=util_block.objective_value >= main_obj.expr)
        else:
            util_block.objective_constr = Constraint(
                expr=util_block.objective_value <= main_obj.expr)
        # Deactivate the original objective and add this new one.
        main_obj.deactivate()
        util_block.objective = Objective(
            expr=util_block.objective_value, sense=main_obj.sense)

        # Add the new variable and constraint to the working lists
        if main_obj.expr.polynomial_degree() not in (1, 0):
            util_block.algebraic_variable_list.append(
                util_block.objective_value)
            #util_blk.continuous_variable_list.append(util_blk.objective_value)
            if hasattr(util_block, 'constraint_list'):
                util_block.constraint_list.append(util_block.objective_constr)
            #util_blk.objective_list.append(util_blk.objective)
            # if util_blk.objective_constr.body.polynomial_degree() in (0, 1):
            #     util_blk.linear_constraint_list.append(
            #         util_blk.objective_constr)
            # else:
            #     util_blk.nonlinear_constraint_list.append(
            #         util_blk.objective_constr)
    
# ESJ: Do we need this? Can it be renamed?
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

    Rounds to Binary/Integer if necessary
    Sets to zero for NonNegativeReals if necessary
    """
    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue  # Skip stale variable values.
        if skip_fixed and v_to.is_fixed():
            continue  # Skip fixed variables.
        try:
            # We don't want to trigger the reset of the global stale
            # indicator, so we will set this variable to be "stale",
            # knowing that set_value will switch it back to "not
            # stale"
            v_to.stale = True
            # NOTE: PEP 2180 changes the var behavior so that domain /
            # bounds violations no longer generate exceptions (and
            # instead log warnings).  This means that the following will
            # always succeed and the ValueError should never be raised.
            v_to.set_value(value(v_from, exception=False), skip_validation=True)
        except ValueError as err:
            err_msg = getattr(err, 'message', str(err))
            var_val = value(v_from)
            rounded_val = int(round(var_val))
            # Check to see if this is just a tolerance issue
            if ignore_integrality and v_to.is_integer():
                v_to.set_value(var_val, skip_validation=True)
            elif v_to.is_integer() and (fabs(var_val - rounded_val) <=
                                        config.integer_tolerance):
                v_to.set_value(rounded_val, skip_validation=True)
            elif abs(var_val) <= config.zero_tolerance and 0 in v_to.domain:
                v_to.set_value(0, skip_validation=True)
            else:
                config.logger.error('Unknown validation domain error setting '
                                    'variable %s', (v_to.name,))
                raise

def fix_discrete_var(var, val, config):
    """Fixes the discrete variable var to val, rounding to the nearest integer
    or not, depending on if rounding is specifed in config and what the integer
    tolerance is."""
    if val is None:
        return
    if var.is_continuous():
        var.set_value(val, skip_validation=True)
    elif (fabs(val - round(val)) > config.integer_tolerance):
        raise ValueError(
            "Integer variable '%s' cannot be fixed to value %s because it "
            "is not within the specified integer tolerance of %s." %
            (var.name, val, config.integer_tolerance))
    else:
        # variable is integer and within tolerance
        if config.round_discrete_vars:
            var.fix(int(round(val)))
        else:
            var.fix(val, skip_validation=True)

@contextmanager
def fix_master_solution_in_subproblem(master_util_block, subproblem_util_block,
                                      config, make_subproblem_continuous=True):
    # fix subproblem Blocks according to the master solution
    fixed = []
    for disjunct, block in zip(master_util_block.disjunct_list,
                               subproblem_util_block.disjunct_list):
        if not disjunct.indicator_var.value:
            block.deactivate()
            block.binary_indicator_var.fix(0)
        else:
            block.binary_indicator_var.fix(1)
            fixed.append(block.name)
    config.logger.debug("Fixed the following Disjuncts to 'True': %s" 
                        % ", ".join(fixed))

    fixed_bools = []
    for master_bool, subprob_bool in zip(
            master_util_block.non_indicator_boolean_variable_list, 
            subproblem_util_block.non_indicator_boolean_variable_list):
        master_binary = master_bool.get_associated_binary()
        subprob_binary = subprob_bool.get_associated_binary()
        val = master_binary.value
        if val is None:
            # If it's None, it's not yet constrained in master problem: make an
            # arbitrary decision for now, and store it in the master problem so
            # the no-good cut will be right.
            master_binary.set_value(1)
            subprob_binary.fix(1)
            bool_val = True
        elif val > 0.5:
            subprob_binary.fix(1)
            bool_val = True
        else:
            subprob_binary.fix(0)
            bool_val = False
        fixed_bools.append("%s = %s" % (subprob_bool.name, bool_val))
    config.logger.debug("Fixed the following Boolean variables: %s"
                        % ", ".join(fixed_bools))

    # Fix subproblem discrete variables according to the master solution
    if make_subproblem_continuous:
        for master_var, subprob_var in zip(
                master_util_block.discrete_variable_list,
                subproblem_util_block.discrete_variable_list):
            # [ESJ 1/24/21]: We don't check if master_var actually has a value
            # here because we are going to have to do that error checking
            # later. This is because the subproblem could have discrete
            # variables that aren't in the master and vice versa since master is
            # linearized, but subproblem is a specific realization of the
            # disjuncts. All this means we don't have enough info to do it here.
            fix_discrete_var(subprob_var, master_var.value, config)

    # ESJ TODO: Why not initialize with the values of the continuous vars too?
    # At least in solve_local_subproblem there was a TODO about that.
            
    yield

    # unfix all subproblem blocks
    for block in subproblem_util_block.disjunct_list:
        block.activate()
        block.binary_indicator_var.unfix()

    # unfix all the formerly-Boolean variables
    for bool_var in subproblem_util_block.non_indicator_boolean_variable_list:
        bool_var.get_associated_binary().unfix()

    # unfix all discrete variables and restore them to their original values
    if make_subproblem_continuous:
        for var in subproblem_util_block.discrete_variable_list:
            subprob_var.fixed = False

    # [ESJ 2/25/22] I think we don't need to reset the values of the continuous
    # variables because we will initialize them based on the master solution
    # before we solve again.

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

# Utility used in cut_generation
def constraints_in_True_disjuncts(model, config):
    """Yield constraints in disjuncts where the indicator value is set or 
    fixed to True."""
    for constr in model.component_data_objects(Constraint):
        yield constr
    observed_disjuncts = ComponentSet()
    for disjctn in model.component_data_objects(Disjunction):
        # get all the disjuncts in the disjunction. Check which ones are True.
        for disj in disjctn.disjuncts:
            if disj in observed_disjuncts:
                continue
            observed_disjuncts.add(disj)
            if fabs(disj.binary_indicator_var.value - 1) \
               <= config.integer_tolerance:
                for constr in disj.component_data_objects(Constraint):
                    yield constr

# ESJ TODO: Can we rename this? Something like time_tracking or time_recording
# or something?
@contextmanager
def time_code(timing_data_obj, code_block_name, is_main_timer=False):
    """Starts timer at entry, stores elapsed time at exit

    If `is_main_timer=True`, the start time is stored in the timing_data_obj,
    allowing calculation of total elapsed time 'on the fly' (e.g. to enforce
    a time limit) using `get_main_elapsed_time(timing_data_obj)`.
    """
    start_time = timing.default_timer()
    if is_main_timer:
        timing_data_obj.main_timer_start_time = start_time
    yield
    elapsed_time = timing.default_timer() - start_time
    prev_time = timing_data_obj.get(code_block_name, 0)
    timing_data_obj[code_block_name] = prev_time + elapsed_time

def get_main_elapsed_time(timing_data_obj):
    """Returns the time since entering the main `time_code` context"""
    current_time = timing.default_timer()
    try:
        return current_time - timing_data_obj.main_timer_start_time
    except AttributeError as e:
        if 'main_timer_start_time' in str(e):
            raise e from AttributeError(
                "You need to be in a 'time_code' context to use "
                "`get_main_elapsed_time()`."
            )

@contextmanager
def lower_logger_level_to(logger, tee=False):
    """Increases logger verbosity by lowering reporting level."""
    level = logging.INFO if tee else None
    handlers = [h for h in logger.handlers]

    if tee: # we want pretty stuff
        logger.handlers.clear()
        logger.propagate = False
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logger.getEffectiveLevel())
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)

    level_changed = False
    if level is not None and logger.getEffectiveLevel() > level:
        # If logger level is higher (less verbose), decrease it
        old_logger_level = logger.level
        logger.setLevel(level)
        if tee:
            sh.setLevel(level)
        level_changed = True

    yield

    if tee:
        logger.handlers.clear()
        for h in handlers:
            logger.addHandler
        logger.propagate = True
    if level_changed:
        logger.setLevel(old_logger_level)
