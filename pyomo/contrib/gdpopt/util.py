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
"""Utility functions and classes for the GDPopt solver."""

from contextlib import contextmanager
import logging
from math import fabs
import sys

from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
    Block,
    Constraint,
    minimize,
    Objective,
    Reals,
    Reference,
    TransformationFactory,
    value,
    Var,
)
from pyomo.core.expr.numvalue import native_types
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
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
                "Loading a SolverResults object with a warning status into model"
            )

    warning_filter = InfeasibleWarningFilter()

    def __enter__(self):
        logger = logging.getLogger('pyomo.core')
        logger.addFilter(self.warning_filter)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        logger = logging.getLogger('pyomo.core')
        logger.removeFilter(self.warning_filter)


def solve_continuous_problem(m, config):
    logger = config.logger
    logger.info('Problem has no discrete decisions.')
    obj = next(m.component_data_objects(Objective, active=True))
    if any(
        c.body.polynomial_degree() not in (1, 0)
        for c in m.component_data_objects(Constraint, active=True, descend_into=Block)
    ) or obj.polynomial_degree() not in (1, 0):
        logger.info(
            "Your model is an NLP (nonlinear program). "
            "Using NLP solver %s to solve." % config.nlp_solver
        )
        results = SolverFactory(config.nlp_solver).solve(m, **config.nlp_solver_args)
        return results
    else:
        logger.info(
            "Your model is an LP (linear program). "
            "Using LP solver %s to solve." % config.mip_solver
        )
        results = SolverFactory(config.mip_solver).solve(m, **config.mip_solver_args)
        return results


def move_nonlinear_objective_to_constraints(util_block, logger):
    m = util_block.parent_block()
    discrete_obj = next(
        m.component_data_objects(Objective, descend_into=True, active=True)
    )
    if discrete_obj.polynomial_degree() in (1, 0):
        # Nothing to move
        return None

    # Move the objective to the constraints if it is nonlinear
    logger.info("Objective is nonlinear. Moving it to constraint set.")

    util_block.objective_value = Var(domain=Reals, initialize=0)
    if mcpp_available():
        mc_obj = McCormick(discrete_obj.expr)
        util_block.objective_value.setub(mc_obj.upper())
        util_block.objective_value.setlb(mc_obj.lower())
    else:
        # Use Pyomo's contrib.fbbt package
        lb, ub = compute_bounds_on_expr(discrete_obj.expr)
        if discrete_obj.sense == minimize:
            util_block.objective_value.setlb(lb)
        else:
            util_block.objective_value.setub(ub)

    if discrete_obj.sense == minimize:
        util_block.objective_constr = Constraint(
            expr=util_block.objective_value >= discrete_obj.expr
        )
    else:
        util_block.objective_constr = Constraint(
            expr=util_block.objective_value <= discrete_obj.expr
        )
    # Deactivate the original objective and add this new one.
    discrete_obj.deactivate()
    util_block.objective = Objective(
        expr=util_block.objective_value, sense=discrete_obj.sense
    )

    # Add the new variable and constraint to the working lists
    util_block.algebraic_variable_list.append(util_block.objective_value)
    if hasattr(util_block, 'constraint_list'):
        util_block.constraint_list.append(util_block.objective_constr)
    # If we moved the objective, return the original in case we want to
    # restore it later
    return discrete_obj


def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)


def copy_var_list_values(
    from_list,
    to_list,
    config,
    skip_stale=False,
    skip_fixed=True,
    ignore_integrality=False,
):
    """Copy variable values from one list to another.

    Rounds to Binary/Integer if necessary
    Sets to zero for NonNegativeReals if necessary
    """
    if ignore_integrality:
        deprecation_warning(
            "The 'ignore_integrality' argument no longer has any functionality.",
            version="6.4.2",
        )

    if len(from_list) != len(to_list):
        raise ValueError('The lengths of from_list and to_list do not match.')

    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue  # Skip stale variable values.
        if skip_fixed and v_to.is_fixed():
            continue  # Skip fixed variables.
        v_to.set_value(value(v_from, exception=False), skip_validation=True)


def fix_discrete_var(var, val, config):
    """Fixes the discrete variable var to val, rounding to the nearest integer
    or not, depending on if rounding is specified in config and what the integer
    tolerance is."""
    if val is None:
        return
    if var.is_continuous():
        var.set_value(val, skip_validation=True)
    elif fabs(val - round(val)) > config.integer_tolerance:
        raise ValueError(
            "Integer variable '%s' cannot be fixed to value %s because it "
            "is not within the specified integer tolerance of %s."
            % (var.name, val, config.integer_tolerance)
        )
    else:
        # variable is integer and within tolerance
        if config.round_discrete_vars:
            var.fix(int(round(val)))
        else:
            var.fix(val, skip_validation=True)


class fix_discrete_solution_in_subproblem(object):
    def __init__(
        self,
        true_disjuncts,
        boolean_var_values,
        integer_var_values,
        subprob_util_block,
        config,
        solver,
    ):
        self.True_disjuncts = true_disjuncts
        self.boolean_var_values = boolean_var_values
        self.discrete_var_values = integer_var_values
        self.subprob_util_block = subprob_util_block
        self.config = config

    def __enter__(self):
        # fix subproblem Blocks according to the discrete problem solution
        fixed = []
        for block in self.subprob_util_block.disjunct_list:
            if block in self.True_disjuncts:
                block.binary_indicator_var.fix(1)
                fixed.append(block.name)
            else:
                block.deactivate()
                block.binary_indicator_var.fix(0)
        self.config.logger.debug(
            "Fixed the following Disjuncts to 'True': %s" % ", ".join(fixed)
        )

        fixed_bools = []
        for subprob_bool, val in zip(
            self.subprob_util_block.non_indicator_boolean_variable_list,
            self.boolean_var_values,
        ):
            subprob_binary = subprob_bool.get_associated_binary()
            if val:
                subprob_binary.fix(1)
            else:
                subprob_binary.fix(0)
            fixed_bools.append("%s = %s" % (subprob_bool.name, val))
        self.config.logger.debug(
            "Fixed the following Boolean variables: %s" % ", ".join(fixed_bools)
        )

        # Fix subproblem discrete variables according to the discrete problem
        # solution
        if self.config.force_subproblem_nlp:
            fixed_discrete = []
            for subprob_var, val in zip(
                self.subprob_util_block.discrete_variable_list, self.discrete_var_values
            ):
                fix_discrete_var(subprob_var, val, self.config)
                fixed_discrete.append("%s = %s" % (subprob_var.name, val))
            self.config.logger.debug(
                "Fixed the following integer variables: "
                "%s" % ", ".join(fixed_discrete)
            )

        # Call the subproblem initialization callback
        self.config.subproblem_initialization_method(
            self.True_disjuncts,
            self.boolean_var_values,
            self.discrete_var_values,
            self.subprob_util_block,
        )

        return self

    def __exit__(self, type, value, traceback):
        # unfix all subproblem blocks
        for block in self.subprob_util_block.disjunct_list:
            block.activate()
            block.binary_indicator_var.unfix()

        # unfix all the formerly-Boolean variables
        for bool_var in self.subprob_util_block.non_indicator_boolean_variable_list:
            bool_var.get_associated_binary().unfix()

        # unfix all discrete variables and restore them to their original values
        if self.config.force_subproblem_nlp:
            for subprob_var in self.subprob_util_block.discrete_variable_list:
                subprob_var.fixed = False

        # [ESJ 2/25/22] We don't need to reset the values of the continuous
        # variables because we will initialize them based on the discrete
        # problem solution before we solve again.


class fix_discrete_problem_solution_in_subproblem(fix_discrete_solution_in_subproblem):
    def __init__(self, discrete_prob_util_block, subproblem_util_block, solver, config):
        self.discrete_prob_util_block = discrete_prob_util_block
        self.subprob_util_block = subproblem_util_block
        self.solver = solver
        self.config = config

    def __enter__(self):
        # fix subproblem Blocks according to the discrete problem solution
        fixed = []
        for disjunct, block in zip(
            self.discrete_prob_util_block.disjunct_list,
            self.subprob_util_block.disjunct_list,
        ):
            if not disjunct.indicator_var.value:
                block.deactivate()
                block.binary_indicator_var.fix(0)
            else:
                block.binary_indicator_var.fix(1)
                fixed.append(block.name)
        self.config.logger.debug(
            "Fixed the following Disjuncts to 'True': %s" % ", ".join(fixed)
        )

        fixed_bools = []
        for discrete_problem_bool, subprob_bool in zip(
            self.discrete_prob_util_block.non_indicator_boolean_variable_list,
            self.subprob_util_block.non_indicator_boolean_variable_list,
        ):
            discrete_problem_binary = discrete_problem_bool.get_associated_binary()
            subprob_binary = subprob_bool.get_associated_binary()
            val = discrete_problem_binary.value
            if val is None:
                # If it's None, it's not yet constrained in discrete problem:
                # make an arbitrary decision for now, and store it in the
                # discrete problem so the no-good cut will be right.
                discrete_problem_binary.set_value(1)
                subprob_binary.fix(1)
                bool_val = True
            elif val > 0.5:
                subprob_binary.fix(1)
                bool_val = True
            else:
                subprob_binary.fix(0)
                bool_val = False
            fixed_bools.append("%s = %s" % (subprob_bool.name, bool_val))
        self.config.logger.debug(
            "Fixed the following Boolean variables: %s" % ", ".join(fixed_bools)
        )

        # Fix subproblem discrete variables according to the discrete problem
        # solution
        if self.config.force_subproblem_nlp:
            fixed_discrete = []
            for discrete_problem_var, subprob_var in zip(
                self.discrete_prob_util_block.discrete_variable_list,
                self.subprob_util_block.discrete_variable_list,
            ):
                # [ESJ 1/24/21]: We don't check if discrete problem_var
                # actually has a value here because we are going to have to do
                # that error checking later. This is because the subproblem
                # could have discrete variables that aren't in the discrete
                # problem and vice versa since discrete problem is linearized,
                # but subproblem is a specific realization of the disjuncts. All
                # this means we don't have enough info to do it here.
                fix_discrete_var(subprob_var, discrete_problem_var.value, self.config)
                fixed_discrete.append(
                    "%s = %s" % (subprob_var.name, discrete_problem_var.value)
                )
            self.config.logger.debug(
                "Fixed the following integer variables: "
                "%s" % ", ".join(fixed_discrete)
            )

        # Call the subproblem initialization callback
        self.config.subproblem_initialization_method(
            self.solver, self.subprob_util_block, self.discrete_prob_util_block
        )

        return self


def is_feasible(model, config):
    """Checks to see if the algebraic model is feasible in its current state.

    Checks variable bounds and active constraints. Not for use with
    untransformed GDP models.

    """
    disj = next(model.component_data_objects(ctype=Disjunct, active=True), None)
    if disj is not None:
        raise NotImplementedError(
            "Found active disjunct %s. "
            "This function is not intended to check "
            "feasibility of disjunctive models, "
            "only transformed subproblems." % disj.name
        )

    config.logger.debug('Checking if model is feasible.')
    for constr in model.component_data_objects(
        ctype=Constraint, active=True, descend_into=True
    ):
        # Check constraint lower bound
        if constr.lower is not None and (
            value(constr.lower) - value(constr.body) >= config.constraint_tolerance
        ):
            config.logger.info(
                '%s: body %s < LB %s'
                % (constr.name, value(constr.body), value(constr.lower))
            )
            return False
        # check constraint upper bound
        if constr.upper is not None and (
            value(constr.body) - value(constr.upper) >= config.constraint_tolerance
        ):
            config.logger.info(
                '%s: body %s > UB %s'
                % (constr.name, value(constr.body), value(constr.upper))
            )
            return False
    for var in model.component_data_objects(ctype=Var, descend_into=True):
        # Check variable lower bound
        if var.has_lb() and value(var.lb) - value(var) >= config.variable_tolerance:
            config.logger.info(
                '%s: value %s < LB %s' % (var.name, value(var), value(var.lb))
            )
            return False
        # Check variable upper bound
        if var.has_ub() and value(var) - value(var.ub) >= config.variable_tolerance:
            config.logger.info(
                '%s: value %s > UB %s' % (var.name, value(var), value(var.ub))
            )
            return False
    config.logger.info('Model is feasible.')
    return True


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
def lower_logger_level_to(logger, level=None, tee=False):
    """Increases logger verbosity by lowering reporting level."""
    if tee:  # we want pretty stuff
        level = logging.INFO  # we need to be at least this verbose for tee to
        # work
        handlers = [h for h in logger.handlers]
        logger.handlers.clear()
        logger.propagate = False
        # Send logging to stdout
        sh = logging.StreamHandler(sys.stdout)
        # set it to the logger level first, we'll change it below if it needs to
        # become more verbose for tee
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
            logger.addHandler(h)
        logger.propagate = True
    if level_changed:
        logger.setLevel(old_logger_level)


def _add_bigm_constraint_to_transformed_model(m, constraint, block):
    """Adds the given constraint to the discrete problem model as if it had
    been on the model originally, before the bigm transformation was called.
    Note this method doesn't actually add the constraint to the model, it just
    takes a constraint that has been added and transforms it.

    Also note that this is not a general method: We know several special
    things in the case of adding OA cuts:
    * No one is going to have a bigm Suffix or arg for this cut--we're
    definitely calculating our own value of M.
    * constraint is for sure a ConstraintData--we don't need to handle anything
    else.
    * We know that we originally called bigm with the default arguments to the
    transformation, so we can assume all of those for this as well. (This is
    part of the reason this *isn't* a general method, what to do about this
    generally is a hard question.)

    Parameters
    ----------
    m: Discrete problem model that has been transformed with bigm.
    constraint: Already-constructed ConstraintData somewhere on m
    block: The block that constraint lives on. This Block may or may not be on
           a Disjunct.
    """
    # Find out it if this constraint really is on a Disjunct. If not, then
    # it's global and we don't actually need to do anything.
    parent_disjunct = block
    if parent_disjunct.ctype is not Disjunct:
        parent_disjunct = _parent_disjunct(block)

    if parent_disjunct is None:
        # the constraint is global, there's nothing to do.
        return

    bigm = TransformationFactory('gdp.bigm')
    # We're fine with default state, but because we're not using apply_to, we
    # need to set it.
    bigm._config = bigm.CONFIG()
    # ESJ: This function doesn't handle ConstraintDatas, and bigm is not
    # sufficiently modular to have a function that does at the moment, so I'm
    # making a Reference to the ComponentData so that it will look like an
    # indexed component for now. If I redesign bigm at some point, then this
    # could be prettier.
    bigm._transform_constraint(Reference(constraint), parent_disjunct, None, [], [])
    # Now get rid of it because this is a class attribute!
    del bigm._config
