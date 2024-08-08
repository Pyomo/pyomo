#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

'''
Utility functions for the PyROS solver
'''

import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
    Constraint,
    Var,
    ConstraintList,
    Objective,
    minimize,
    Expression,
    ConcreteModel,
    maximize,
    Block,
    Param,
)
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.repn.plugins.nl_writer as pyomo_nl_writer
import pyomo.repn.ampl as pyomo_ampl_repn
from pyomo.core.expr.visitor import (
    identify_variables,
    identify_mutable_parameters,
    replace_expressions,
)
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory

import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted


# Tolerances used in the code
PARAM_IS_CERTAIN_REL_TOL = 1e-4
PARAM_IS_CERTAIN_ABS_TOL = 0
COEFF_MATCH_REL_TOL = 1e-6
COEFF_MATCH_ABS_TOL = 0
ABS_CON_CHECK_FEAS_TOL = 1e-5
TIC_TOC_SOLVE_TIME_ATTR = "pyros_tic_toc_time"
DEFAULT_LOGGER_NAME = "pyomo.contrib.pyros"


class TimingData:
    """
    PyROS solver timing data object.

    Implemented as a wrapper around `common.timing.HierarchicalTimer`,
    with added functionality for enforcing a standardized
    hierarchy of identifiers.

    Attributes
    ----------
    hierarchical_timer_full_ids : set of str
        (Class attribute.) Valid identifiers for use with
        the encapsulated hierarchical timer.
    """

    hierarchical_timer_full_ids = {
        "main",
        "main.preprocessing",
        "main.master_feasibility",
        "main.master",
        "main.dr_polishing",
        "main.local_separation",
        "main.global_separation",
    }

    def __init__(self):
        """Initialize self (see class docstring)."""
        self._hierarchical_timer = HierarchicalTimer()

    def __str__(self):
        """
        String representation of `self`. Currently
        returns the string representation of `self.hierarchical_timer`.

        Returns
        -------
        str
            String representation.
        """
        return self._hierarchical_timer.__str__()

    def _validate_full_identifier(self, full_identifier):
        """
        Validate identifier for hierarchical timer.

        Parameters
        ----------
        full_identifier : str
            Identifier to validate.

        Raises
        ------
        ValueError
            If identifier not in `TimingData.hierarchical_timer_full_ids`.
        """
        if full_identifier not in self.hierarchical_timer_full_ids:
            raise ValueError(
                "PyROS timing data object does not support timing ID: "
                f"{full_identifier}."
            )

    def start_timer(self, full_identifier):
        """
        Start timer for `self.hierarchical_timer`.

        Parameters
        ----------
        full_identifier : str
            Full identifier for the timer to be started.
            Must be an entry of
            `TimingData.hierarchical_timer_full_ids`.
        """
        self._validate_full_identifier(full_identifier)
        identifier = full_identifier.split(".")[-1]
        return self._hierarchical_timer.start(identifier=identifier)

    def stop_timer(self, full_identifier):
        """
        Stop timer for `self.hierarchical_timer`.

        Parameters
        ----------
        full_identifier : str
            Full identifier for the timer to be stopped.
            Must be an entry of
            `TimingData.hierarchical_timer_full_ids`.
        """
        self._validate_full_identifier(full_identifier)
        identifier = full_identifier.split(".")[-1]
        return self._hierarchical_timer.stop(identifier=identifier)

    def get_total_time(self, full_identifier):
        """
        Get total time spent with identifier active.

        Parameters
        ----------
        full_identifier : str
            Full identifier for the timer of interest.

        Returns
        -------
        float
            Total time spent with identifier active.
        """
        return self._hierarchical_timer.get_total_time(identifier=full_identifier)

    def get_main_elapsed_time(self):
        """
        Get total time elapsed for main timer of
        the HierarchicalTimer contained in self.

        Returns
        -------
        float
            Total elapsed time.

        Note
        ----
        This method is meant for use while the main timer is active.
        Otherwise, use ``self.get_total_time("main")``.
        """
        # clean?
        return self._hierarchical_timer.timers["main"].tic_toc.toc(
            msg=None, delta=False
        )


'''Code borrowed from gdpopt: time_code, get_main_elapsed_time, a_logger.'''


@contextmanager
def time_code(timing_data_obj, code_block_name, is_main_timer=False):
    """
    Starts timer at entry, stores elapsed time at exit.

    Parameters
    ----------
    timing_data_obj : TimingData
        Timing data object.
    code_block_name : str
        Name of code block being timed.

    If `is_main_timer=True`, the start time is stored in the timing_data_obj,
    allowing calculation of total elapsed time 'on the fly' (e.g. to enforce
    a time limit) using `get_main_elapsed_time(timing_data_obj)`.
    """
    # initialize tic toc timer
    timing_data_obj.start_timer(code_block_name)

    start_time = timeit.default_timer()
    if is_main_timer:
        timing_data_obj.main_timer_start_time = start_time
    yield
    timing_data_obj.stop_timer(code_block_name)


def get_main_elapsed_time(timing_data_obj):
    """Returns the time since entering the main `time_code` context"""
    return timing_data_obj.get_main_elapsed_time()


def adjust_solver_time_settings(timing_data_obj, solver, config):
    """
    Adjust maximum time allowed for subordinate solver, based
    on total PyROS solver elapsed time up to this point.

    Parameters
    ----------
    timing_data_obj : Bunch
        PyROS timekeeper.
    solver : solver type
        Subordinate solver for which to adjust the max time setting.
    config : ConfigDict
        PyROS solver config.

    Returns
    -------
    original_max_time_setting : float or None
        If IPOPT or BARON is used, a float is returned.
        If GAMS is used, the ``options.add_options`` attribute
        of ``solver`` is returned.
        Otherwise, None is returned.
    custom_setting_present : bool or None
        If IPOPT or BARON is used, True if the max time is
        specified, False otherwise.
        If GAMS is used, True if the attribute ``options.add_options``
        is not None, False otherwise.
        If ``config.time_limit`` is None, then None is returned.

    Note
    ----
    (1) Adjustment only supported for GAMS, BARON, and IPOPT
        interfaces. This routine can be generalized to other solvers
        after a generic Pyomo interface to the time limit setting
        is introduced.
    (2) For IPOPT and BARON, the CPU time limit,
        rather than the wallclock time limit, may be adjusted,
        as there may be no means by which to specify the wall time
        limit explicitly.
    (3) For GAMS, we adjust the time limit through the GAMS Reslim
        option. However, this may be overridden by any user
        specifications included in a GAMS optfile, which may be
        difficult to track down.
    (4) To ensure the time limit is specified to a strictly
        positive value, the time limit is adjusted to a value of
        at least 1 second.
    """
    # in case there is no time remaining: we set time limit
    # to a minimum of 1s, as some solvers require a strictly
    # positive time limit
    time_limit_buffer = 1

    if config.time_limit is not None:
        time_remaining = config.time_limit - get_main_elapsed_time(timing_data_obj)
        if isinstance(solver, type(SolverFactory("gams", solver_io="shell"))):
            original_max_time_setting = solver.options["add_options"]
            custom_setting_present = "add_options" in solver.options

            # note: our time limit will be overridden by any
            #       time limits specified by the user through a
            #       GAMS optfile, but tracking down the optfile
            #       and/or the GAMS subsolver specific option
            #       is more difficult
            reslim_str = "option reslim=" f"{max(time_limit_buffer, time_remaining)};"
            if isinstance(solver.options["add_options"], list):
                solver.options["add_options"].append(reslim_str)
            else:
                solver.options["add_options"] = [reslim_str]
        else:
            # determine name of option to adjust
            if isinstance(solver, SolverFactory.get_class("baron")):
                options_key = "MaxTime"
            elif isinstance(solver, SolverFactory.get_class("ipopt")):
                options_key = (
                    # IPOPT 3.14.0+ added support for specifying
                    # wall time limit explicitly; this is preferred
                    # over CPU time limit
                    "max_wall_time"
                    if solver.version() >= (3, 14, 0, 0)
                    else "max_cpu_time"
                )
            elif isinstance(solver, SolverFactory.get_class("scip")):
                options_key = "limits/time"
            else:
                options_key = None

            if options_key is not None:
                custom_setting_present = options_key in solver.options
                original_max_time_setting = solver.options[options_key]

                # account for elapsed time remaining and
                # original time limit setting.
                # if no original time limit is set, then we assume
                # there is no time limit, rather than tracking
                # down the solver-specific default
                orig_max_time = (
                    float("inf")
                    if original_max_time_setting is None
                    else original_max_time_setting
                )
                solver.options[options_key] = min(
                    max(time_limit_buffer, time_remaining), orig_max_time
                )
            else:
                custom_setting_present = False
                original_max_time_setting = None
                config.progress_logger.warning(
                    "Subproblem time limit setting not adjusted for "
                    f"subsolver of type:\n    {type(solver)}.\n"
                    "    PyROS time limit may not be honored "
                )

        return original_max_time_setting, custom_setting_present
    else:
        return None, None


def revert_solver_max_time_adjustment(
    solver, original_max_time_setting, custom_setting_present, config
):
    """
    Revert solver `options` attribute to its state prior to a
    time limit adjustment performed via
    the routine `adjust_solver_time_settings`.

    Parameters
    ----------
    solver : solver type
        Solver of interest.
    original_max_time_setting : float, list, or None
        Original solver settings. Type depends on the
        solver type.
    custom_setting_present : bool or None
        Was the max time, or other custom solver settings,
        specified prior to the adjustment?
        Can be None if ``config.time_limit`` is None.
    config : ConfigDict
        PyROS solver config.
    """
    if config.time_limit is not None:
        assert isinstance(custom_setting_present, bool)

        # determine name of option to adjust
        if isinstance(solver, type(SolverFactory("gams", solver_io="shell"))):
            options_key = "add_options"
        elif isinstance(solver, SolverFactory.get_class("baron")):
            options_key = "MaxTime"
        elif isinstance(solver, SolverFactory.get_class("ipopt")):
            options_key = (
                # IPOPT 3.14.0+ added support for specifying
                # wall time limit explicitly; this is preferred
                # over CPU time limit
                "max_wall_time"
                if solver.version() >= (3, 14, 0, 0)
                else "max_cpu_time"
            )
        elif isinstance(solver, SolverFactory.get_class("scip")):
            options_key = "limits/time"
        else:
            options_key = None

        if options_key is not None:
            if custom_setting_present:
                # restore original setting
                solver.options[options_key] = original_max_time_setting

                # if GAMS solver used, need to remove the last entry
                # of 'add_options', which contains the max time setting
                # added by PyROS
                if isinstance(solver, type(SolverFactory("gams", solver_io="shell"))):
                    solver.options[options_key].pop()
            else:
                delattr(solver.options, options_key)


class PreformattedLogger(logging.Logger):
    """
    A specialized logger object designed to cast log messages
    to Pyomo `Preformatted` objects prior to logging the messages.
    Useful for circumventing the formatters of the standard Pyomo
    logger in the event an instance is a descendant of the Pyomo
    logger.
    """

    def critical(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.CRITICAL`.
        """
        return super(PreformattedLogger, self).critical(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def error(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.ERROR`.
        """
        return super(PreformattedLogger, self).error(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def warning(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.WARNING`.
        """
        return super(PreformattedLogger, self).warning(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def info(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.INFO`.
        """
        return super(PreformattedLogger, self).info(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def debug(self, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with severity
        `logging.DEBUG`.
        """
        return super(PreformattedLogger, self).debug(
            Preformatted(msg % args if args else msg), **kwargs
        )

    def log(self, level, msg, *args, **kwargs):
        """
        Preformat and log ``msg % args`` with integer
        severity `level`.
        """
        return super(PreformattedLogger, self).log(
            level, Preformatted(msg % args if args else msg), **kwargs
        )


def setup_pyros_logger(name=DEFAULT_LOGGER_NAME):
    """
    Set up pyros logger.
    """
    # default logger: INFO level, with preformatted messages
    current_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(PreformattedLogger)
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    logging.setLoggerClass(current_logger_class)

    return logger


class pyrosTerminationCondition(Enum):
    """Enumeration of all possible PyROS termination conditions."""

    robust_feasible = 0
    """Final solution is robust feasible."""

    robust_optimal = 1
    """Final solution is robust optimal."""

    robust_infeasible = 2
    """Problem is robust infeasible."""

    max_iter = 3
    """Maximum number of GRCS iteration reached."""

    subsolver_error = 4
    """Subsolver(s) provided could not solve a subproblem to
    an acceptable termination status."""

    time_out = 5
    """Maximum allowable time exceeded."""

    @property
    def message(self):
        """
        str : Message associated with a given PyROS
        termination condition.
        """
        message_dict = {
            self.robust_optimal: "Robust optimal solution identified.",
            self.robust_feasible: "Robust feasible solution identified.",
            self.robust_infeasible: "Problem is robust infeasible.",
            self.time_out: "Maximum allowable time exceeded.",
            self.max_iter: "Maximum number of iterations reached.",
            self.subsolver_error: (
                "Subordinate optimizer(s) could not solve a subproblem "
                "to an acceptable status."
            ),
        }
        return message_dict[self]


class SeparationStrategy(Enum):
    all_violations = auto()
    max_violation = auto()


class SolveMethod(Enum):
    local_solve = auto()
    global_solve = auto()


class ObjectiveType(Enum):
    worst_case = auto()
    nominal = auto()


def recast_to_min_obj(model, obj):
    """
    Recast model objective to a minimization objective, as necessary.

    Parameters
    ----------
    model : ConcreteModel
        Model of interest.
    obj : ScalarObjective
        Objective of interest.
    """
    if obj.sense is not minimize:
        if isinstance(obj.expr, SumExpression):
            # ensure additive terms in objective
            # are split in accordance with user declaration
            obj.expr = sum(-term for term in obj.expr.args)
        else:
            obj.expr = -obj.expr
        obj.sense = minimize


def turn_bounds_to_constraints(variable, model, config=None):
    '''
    Turn the variable in question's "bounds" into direct inequality constraints on the model.
    :param variable: the variable with bounds to be turned to None and made into constraints.
    :param model: the model in which the variable resides
    :param config: solver config
    :return: the list of inequality constraints that are the bounds
    '''
    lb, ub = variable.lower, variable.upper
    if variable.domain is not Reals:
        variable.domain = Reals

    if isinstance(lb, NPV_MaxExpression):
        lb_args = lb.args
    else:
        lb_args = (lb,)

    if isinstance(ub, NPV_MinExpression):
        ub_args = ub.args
    else:
        ub_args = (ub,)

    count = 0
    for arg in lb_args:
        if arg is not None:
            name = unique_component_name(
                model, variable.name + f"_lower_bound_con_{count}"
            )
            model.add_component(name, Constraint(expr=arg - variable <= 0))
            count += 1
            variable.setlb(None)

    count = 0
    for arg in ub_args:
        if arg is not None:
            name = unique_component_name(
                model, variable.name + f"_upper_bound_con_{count}"
            )
            model.add_component(name, Constraint(expr=variable - arg <= 0))
            count += 1
            variable.setub(None)


def get_time_from_solver(results):
    """
    Obtain solver time from a Pyomo `SolverResults` object.

    Returns
    -------
    : float
        Solver time. May be CPU time or elapsed time,
        depending on the solver. If no time attribute
        is found, then `float("nan")` is returned.

    NOTE
    ----
    This method attempts to access solver time through the
    attributes of `results.solver` in the following order
    of precedence:

    1) Attribute with name ``pyros.util.TIC_TOC_SOLVE_TIME_ATTR``.
       This attribute is an estimate of the elapsed solve time
       obtained using the Pyomo `TicTocTimer` at the point the
       solver from which the results object is derived was invoked.
       Preferred over other time attributes, as other attributes
       may be in CPUs, and for purposes of evaluating overhead
       time, we require wall s.
    2) `'user_time'` if the results object was returned by a GAMS
       solver, `'time'` otherwise.
    """
    solver_name = getattr(results.solver, "name", None)

    # is this sufficient to confirm GAMS solver used?
    from_gams = solver_name is not None and str(solver_name).startswith("GAMS ")
    time_attr_name = "user_time" if from_gams else "time"
    for attr_name in [TIC_TOC_SOLVE_TIME_ATTR, time_attr_name]:
        solve_time = getattr(results.solver, attr_name, None)
        if solve_time is not None:
            break

    return float("nan") if solve_time is None else solve_time


def add_bounds_for_uncertain_parameters(model, config):
    '''
    This function solves a set of optimization problems to determine bounds on the uncertain parameters
    given the uncertainty set description. These bounds will be added as additional constraints to the uncertainty_set_constr
    constraint. Should only be called once set_as_constraint() has been called on the separation_model object.
    :param separation_model: the model on which to add the bounds
    :param config: solver config
    :return:
    '''
    # === Determine bounds on all uncertain params
    uncertain_param_bounds = []
    bounding_model = ConcreteModel()
    bounding_model.util = Block()
    bounding_model.util.uncertain_param_vars = IndexedVar(
        model.util.uncertain_param_vars.index_set()
    )
    for tup in model.util.uncertain_param_vars.items():
        bounding_model.util.uncertain_param_vars[tup[0]].set_value(
            tup[1].value, skip_validation=True
        )

    bounding_model.add_component(
        "uncertainty_set_constraint",
        config.uncertainty_set.set_as_constraint(
            uncertain_params=bounding_model.util.uncertain_param_vars,
            model=bounding_model,
            config=config,
        ),
    )

    for idx, param in enumerate(
        list(bounding_model.util.uncertain_param_vars.values())
    ):
        bounding_model.add_component(
            "lb_obj_" + str(idx), Objective(expr=param, sense=minimize)
        )
        bounding_model.add_component(
            "ub_obj_" + str(idx), Objective(expr=param, sense=maximize)
        )

    for o in bounding_model.component_data_objects(Objective):
        o.deactivate()

    for i in range(len(bounding_model.util.uncertain_param_vars)):
        bounds = []
        for limit in ("lb", "ub"):
            getattr(bounding_model, limit + "_obj_" + str(i)).activate()
            res = config.global_solver.solve(bounding_model, tee=False)
            bounds.append(bounding_model.util.uncertain_param_vars[i].value)
            getattr(bounding_model, limit + "_obj_" + str(i)).deactivate()
        uncertain_param_bounds.append(bounds)

    # === Add bounds as constraints to uncertainty_set_constraint ConstraintList
    for idx, bound in enumerate(uncertain_param_bounds):
        model.util.uncertain_param_vars[idx].setlb(bound[0])
        model.util.uncertain_param_vars[idx].setub(bound[1])

    return


def transform_to_standard_form(model):
    """
    Recast all model inequality constraints of the form `a <= g(v)` (`<= b`)
    to the 'standard' form `a - g(v) <= 0` (and `g(v) - b <= 0`),
    in which `v` denotes all model variables and `a` and `b` are
    contingent on model parameters.

    Parameters
    ----------
    model : ConcreteModel
        The model to search for constraints. This will descend into all
        active Blocks and sub-Blocks as well.

    Note
    ----
    If `a` and `b` are identical and the constraint is not classified as an
    equality (i.e. the `equality` attribute of the constraint object
    is `False`), then the constraint is recast to the equality `g(v) == a`.
    """
    # Note: because we will be adding / modifying the number of
    # constraints, we want to resolve the generator to a list before
    # starting.
    cons = list(
        model.component_data_objects(Constraint, descend_into=True, active=True)
    )
    for con in cons:
        if not con.equality:
            has_lb = con.lower is not None
            has_ub = con.upper is not None

            if has_lb and has_ub:
                if con.lower is con.upper:
                    # recast as equality Constraint
                    con.set_value(con.lower == con.body)
                else:
                    # range inequality; split into two Constraints.
                    uniq_name = unique_component_name(model, con.name + '_lb')
                    model.add_component(
                        uniq_name, Constraint(expr=con.lower - con.body <= 0)
                    )
                    con.set_value(con.body - con.upper <= 0)
            elif has_lb:
                # not in standard form; recast.
                con.set_value(con.lower - con.body <= 0)
            elif has_ub:
                # move upper bound to body.
                con.set_value(con.body - con.upper <= 0)
            else:
                # unbounded constraint: deactivate
                con.deactivate()


def get_vars_from_component(block, ctype):
    """Determine all variables used in active components within a block.

    Parameters
    ----------
    block: Block
        The block to search for components.  This is a recursive
        generator and will descend into any active sub-Blocks as well.
    ctype:  class
        The component type (typically either :py:class:`Constraint` or
        :py:class:`Objective` to search for).

    """

    return get_vars_from_components(block, ctype, active=True, descend_into=True)


def replace_uncertain_bounds_with_constraints(model, uncertain_params):
    """
    For variables of which the bounds are dependent on the parameters
    in the list `uncertain_params`, remove the bounds and add
    explicit variable bound inequality constraints.

    :param model: Model in which to make the bounds/constraint replacements
    :type model: class:`pyomo.core.base.PyomoModel.ConcreteModel`
    :param uncertain_params: List of uncertain model parameters
    :type uncertain_params: list
    """
    uncertain_param_set = ComponentSet(uncertain_params)

    # component for explicit inequality constraints
    uncertain_var_bound_constrs = ConstraintList()
    model.add_component(
        unique_component_name(model, 'uncertain_var_bound_cons'),
        uncertain_var_bound_constrs,
    )

    # get all variables in active objective and constraint expression(s)
    vars_in_cons = ComponentSet(get_vars_from_component(model, Constraint))
    vars_in_obj = ComponentSet(get_vars_from_component(model, Objective))

    for v in vars_in_cons | vars_in_obj:
        # get mutable parameters in variable bounds expressions
        ub = v.upper
        mutable_params_ub = ComponentSet(identify_mutable_parameters(ub))
        lb = v.lower
        mutable_params_lb = ComponentSet(identify_mutable_parameters(lb))

        # add explicit inequality constraint(s), remove variable bound(s)
        if mutable_params_ub & uncertain_param_set:
            if type(ub) is NPV_MinExpression:
                upper_bounds = ub.args
            else:
                upper_bounds = (ub,)
            for u_bnd in upper_bounds:
                uncertain_var_bound_constrs.add(v - u_bnd <= 0)
            v.setub(None)
        if mutable_params_lb & uncertain_param_set:
            if type(ub) is NPV_MaxExpression:
                lower_bounds = lb.args
            else:
                lower_bounds = (lb,)
            for l_bnd in lower_bounds:
                uncertain_var_bound_constrs.add(l_bnd - v <= 0)
            v.setlb(None)


def check_components_descended_from_model(model, components, components_name, config):
    """
    Check all members in a provided sequence of Pyomo component
    objects are descended from a given ConcreteModel object.

    Parameters
    ----------
    model : ConcreteModel
        Model from which components should all be descended.
    components : Iterable of Component
        Components of interest.
    components_name : str
        Brief description or name for the sequence of components.
        Used for constructing error messages.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one entry of `components` is not descended
        from `model`.
    """
    components_not_in_model = [comp for comp in components if comp.model() is not model]
    if components_not_in_model:
        comp_names_str = "\n ".join(
            f"{comp.name!r}, from model with name {comp.model().name!r}"
            for comp in components_not_in_model
        )
        config.progress_logger.error(
            f"The following {components_name} "
            "are not descended from the "
            f"input deterministic model with name {model.name!r}:\n "
            f"{comp_names_str}"
        )
        raise ValueError(
            f"Found entries of {components_name} "
            "not descended from input model. "
            "Check logger output messages."
        )


def get_state_vars(blk, first_stage_variables, second_stage_variables):
    """
    Get state variables of a modeling block.

    The state variables with respect to `blk` are the unfixed
    `VarData` objects participating in the active objective
    or constraints descended from `blk` which are not
    first-stage variables or second-stage variables.

    Parameters
    ----------
    blk : ScalarBlock
        Block of interest.
    first_stage_variables : Iterable of VarData
        First-stage variables.
    second_stage_variables : Iterable of VarData
        Second-stage variables.

    Yields
    ------
    VarData
        State variable.
    """
    dof_var_set = ComponentSet(first_stage_variables) | ComponentSet(
        second_stage_variables
    )
    for var in get_vars_from_component(blk, (Objective, Constraint)):
        is_state_var = not var.fixed and var not in dof_var_set
        if is_state_var:
            yield var


def check_variables_continuous(model, vars, config):
    """
    Check that all DOF and state variables of the model
    are continuous.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one variable is found to not be continuous.

    Note
    ----
    A variable is considered continuous if the `is_continuous()`
    method returns True.
    """
    non_continuous_vars = [var for var in vars if not var.is_continuous()]
    if non_continuous_vars:
        non_continuous_vars_str = "\n ".join(
            f"{var.name!r}" for var in non_continuous_vars
        )
        config.progress_logger.error(
            f"The following Vars of model with name {model.name!r} "
            f"are non-continuous:\n {non_continuous_vars_str}\n"
            "Ensure all model variables passed to PyROS solver are continuous."
        )
        raise ValueError(
            f"Model with name {model.name!r} contains non-continuous Vars."
        )


def validate_model(model, config):
    """
    Validate deterministic model passed to PyROS solver.

    Parameters
    ----------
    model : ConcreteModel
        Deterministic model. Should have only one active Objective.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    ComponentSet
        The variables participating in the active Objective
        and Constraint expressions of `model`.

    Raises
    ------
    TypeError
        If model is not of type ConcreteModel.
    ValueError
        If model does not have exactly one active Objective
        component.
    """
    # note: only support ConcreteModel. no support for Blocks
    if not isinstance(model, ConcreteModel):
        raise TypeError(
            f"Model should be of type {ConcreteModel.__name__}, "
            f"but is of type {type(model).__name__}."
        )

    # active objectives check
    active_objs_list = list(
        model.component_data_objects(Objective, active=True, descend_into=True)
    )
    if len(active_objs_list) != 1:
        raise ValueError(
            "Expected model with exactly 1 active objective, but "
            f"model provided has {len(active_objs_list)}."
        )


def validate_variable_partitioning(model, config):
    """
    Check that partitioning of the first-stage variables,
    second-stage variables, and uncertain parameters
    is valid.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    list of VarData
        State variables of the model.

    Raises
    ------
    ValueError
        If first-stage variables and second-stage variables
        overlap, or there are no first-stage variables
        and no second-stage variables.
    """
    # at least one DOF required
    if not config.first_stage_variables and not config.second_stage_variables:
        raise ValueError(
            "Arguments `first_stage_variables` and "
            "`second_stage_variables` are both empty lists."
        )

    # ensure no overlap between DOF var sets
    overlapping_vars = ComponentSet(config.first_stage_variables) & ComponentSet(
        config.second_stage_variables
    )
    if overlapping_vars:
        overlapping_var_list = "\n ".join(f"{var.name!r}" for var in overlapping_vars)
        config.progress_logger.error(
            "The following Vars were found in both `first_stage_variables`"
            f"and `second_stage_variables`:\n {overlapping_var_list}"
            "\nEnsure no Vars are included in both arguments."
        )
        raise ValueError(
            "Arguments `first_stage_variables` and `second_stage_variables` "
            "contain at least one common Var object."
        )

    state_vars = list(
        get_state_vars(
            model,
            first_stage_variables=config.first_stage_variables,
            second_stage_variables=config.second_stage_variables,
        )
    )
    var_type_list_map = {
        "first-stage variables": config.first_stage_variables,
        "second-stage variables": config.second_stage_variables,
        "state variables": state_vars,
    }
    for desc, vars in var_type_list_map.items():
        check_components_descended_from_model(
            model=model, components=vars, components_name=desc, config=config
        )

    all_vars = config.first_stage_variables + config.second_stage_variables + state_vars
    check_variables_continuous(model, all_vars, config)

    return state_vars


def validate_uncertainty_specification(model, config):
    """
    Validate specification of uncertain parameters and uncertainty
    set.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one of the following holds:

        - dimension of uncertainty set does not equal number of
          uncertain parameters
        - uncertainty set `is_valid()` method does not return
          true.
        - nominal parameter realization is not in the uncertainty set.
    """
    check_components_descended_from_model(
        model=model,
        components=config.uncertain_params,
        components_name="uncertain parameters",
        config=config,
    )

    if len(config.uncertain_params) != config.uncertainty_set.dim:
        raise ValueError(
            "Length of argument `uncertain_params` does not match dimension "
            "of argument `uncertainty_set` "
            f"({len(config.uncertain_params)} != {config.uncertainty_set.dim})."
        )

    # validate uncertainty set
    if not config.uncertainty_set.is_valid(config=config):
        raise ValueError(
            f"Uncertainty set {config.uncertainty_set} is invalid, "
            "as it is either empty or unbounded."
        )

    # fill-in nominal point as necessary, if not provided.
    # otherwise, check length matches uncertainty dimension
    if not config.nominal_uncertain_param_vals:
        config.nominal_uncertain_param_vals = [
            value(param, exception=True) for param in config.uncertain_params
        ]
    elif len(config.nominal_uncertain_param_vals) != len(config.uncertain_params):
        raise ValueError(
            "Lengths of arguments `uncertain_params` and "
            "`nominal_uncertain_param_vals` "
            "do not match "
            f"({len(config.uncertain_params)} != "
            f"{len(config.nominal_uncertain_param_vals)})."
        )

    # uncertainty set should contain nominal point
    nominal_point_in_set = config.uncertainty_set.point_in_set(
        point=config.nominal_uncertain_param_vals
    )
    if not nominal_point_in_set:
        raise ValueError(
            "Nominal uncertain parameter realization "
            f"{config.nominal_uncertain_param_vals} "
            "is not a point in the uncertainty set "
            f"{config.uncertainty_set!r}."
        )


def validate_separation_problem_options(model, config):
    """
    Validate separation problem arguments to the PyROS solver.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If options `bypass_local_separation` and
        `bypass_global_separation` are set to False.
    """
    if config.bypass_local_separation and config.bypass_global_separation:
        raise ValueError(
            "Arguments `bypass_local_separation` "
            "and `bypass_global_separation` "
            "cannot both be True."
        )


def validate_pyros_inputs(model, config):
    """
    Perform advanced validation of PyROS solver arguments.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.
    """
    validate_model(model, config)
    state_vars = validate_variable_partitioning(model, config)
    validate_uncertainty_specification(model, config)
    validate_separation_problem_options(model, config)

    return state_vars


def substitute_ssv_in_dr_constraints(model, constraint):
    '''
    Generate the standard_repn for the dr constraints. Generate new expression with replace_expression to ignore
    the ssv component.
    Then, replace_expression with substitution_map between ssv and the new expression.
    Deactivate or del_component the original dr equation.
    Then, return modified model and do coefficient matching as normal.
    :param model: the working_model
    :param constraint: an equality constraint from the working model identified to be of the form h(x,z,q) = 0.
    :return:
    '''
    dr_eqns = model.util.decision_rule_eqns
    fsv = ComponentSet(model.util.first_stage_variables)
    if not hasattr(model, "dr_substituted_constraints"):
        model.dr_substituted_constraints = ConstraintList()

    substitution_map = {}
    for eqn in dr_eqns:
        repn = generate_standard_repn(eqn.body, compute_values=False)
        new_expression = 0
        map_linear_coeff_to_var = [
            x
            for x in zip(repn.linear_coefs, repn.linear_vars)
            if x[1] in ComponentSet(fsv)
        ]
        map_quad_coeff_to_var = [
            x
            for x in zip(repn.quadratic_coefs, repn.quadratic_vars)
            if x[1] in ComponentSet(fsv)
        ]
        if repn.linear_coefs:
            for coeff, var in map_linear_coeff_to_var:
                new_expression += coeff * var
        if repn.quadratic_coefs:
            for coeff, var in map_quad_coeff_to_var:
                new_expression += coeff * var[0] * var[1]  # var here is a 2-tuple

        substitution_map[id(repn.linear_vars[-1])] = new_expression

    model.dr_substituted_constraints.add(
        replace_expressions(expr=constraint.lower, substitution_map=substitution_map)
        == replace_expressions(expr=constraint.body, substitution_map=substitution_map)
    )

    # === Delete the original constraint
    model.del_component(constraint.name)

    return model.dr_substituted_constraints[
        max(model.dr_substituted_constraints.keys())
    ]


def is_certain_parameter(uncertain_param_index, config):
    '''
    If an uncertain parameter's inferred LB and UB are within a relative tolerance,
    then the parameter is considered certain.
    :param uncertain_param_index: index of the parameter in the config.uncertain_params list
    :param config: solver config
    :return: True if param is effectively "certain," else return False
    '''
    if config.uncertainty_set.parameter_bounds:
        param_bounds = config.uncertainty_set.parameter_bounds[uncertain_param_index]
        return math.isclose(
            a=param_bounds[0],
            b=param_bounds[1],
            rel_tol=PARAM_IS_CERTAIN_REL_TOL,
            abs_tol=PARAM_IS_CERTAIN_ABS_TOL,
        )
    else:
        return False  # cannot be determined without bounds


def coefficient_matching(model, constraint, uncertain_params, config):
    '''
    :param model: master problem model
    :param constraint: the constraint from the master problem model
    :param uncertain_params: the list of uncertain parameters
    :param first_stage_variables: the list of effective first-stage variables (includes ssv if decision_rule_order = 0)
    :return: True if the coefficient matching was successful, False if its proven robust_infeasible due to
             constraints of the form 1 == 0
    '''
    # === Returned flags
    successful_matching = True
    robust_infeasible = False

    # === Efficiency for q_LB = q_UB
    actual_uncertain_params = []

    for i in range(len(uncertain_params)):
        if not is_certain_parameter(uncertain_param_index=i, config=config):
            actual_uncertain_params.append(uncertain_params[i])

    # === Add coefficient matching constraint list
    if not hasattr(model, "coefficient_matching_constraints"):
        model.coefficient_matching_constraints = ConstraintList()
    if not hasattr(model, "swapped_constraints"):
        model.swapped_constraints = ConstraintList()

    variables_in_constraint = ComponentSet(identify_variables(constraint.expr))
    params_in_constraint = ComponentSet(identify_mutable_parameters(constraint.expr))
    first_stage_variables = model.util.first_stage_variables
    second_stage_variables = model.util.second_stage_variables

    # === Determine if we need to do DR expression/ssv substitution to
    #     make h(x,z,q) == 0 into h(x,d,q) == 0 (which is just h(x,q) == 0)
    if all(
        v in ComponentSet(first_stage_variables) for v in variables_in_constraint
    ) and any(q in ComponentSet(actual_uncertain_params) for q in params_in_constraint):
        # h(x, q) == 0
        pass
    elif all(
        v in ComponentSet(first_stage_variables + second_stage_variables)
        for v in variables_in_constraint
    ) and any(q in ComponentSet(actual_uncertain_params) for q in params_in_constraint):
        constraint = substitute_ssv_in_dr_constraints(
            model=model, constraint=constraint
        )

        variables_in_constraint = ComponentSet(identify_variables(constraint.expr))
        params_in_constraint = ComponentSet(
            identify_mutable_parameters(constraint.expr)
        )
    else:
        pass

    if all(
        v in ComponentSet(first_stage_variables) for v in variables_in_constraint
    ) and any(q in ComponentSet(actual_uncertain_params) for q in params_in_constraint):
        # Swap param objects for variable objects in this constraint
        model.param_set = []
        for i in range(len(list(variables_in_constraint))):
            # Initialize Params to non-zero value due to standard_repn bug
            model.add_component("p_%s" % i, Param(initialize=1, mutable=True))
            model.param_set.append(getattr(model, "p_%s" % i))

        model.variable_set = []
        for i in range(len(list(actual_uncertain_params))):
            model.add_component("x_%s" % i, Var(initialize=1))
            model.variable_set.append(getattr(model, "x_%s" % i))

        original_var_to_param_map = list(
            zip(list(variables_in_constraint), model.param_set)
        )
        original_param_to_vap_map = list(
            zip(list(actual_uncertain_params), model.variable_set)
        )

        var_to_param_substitution_map_forward = {}
        # Separation problem initialized to nominal uncertain parameter values
        for var, param in original_var_to_param_map:
            var_to_param_substitution_map_forward[id(var)] = param

        param_to_var_substitution_map_forward = {}
        # Separation problem initialized to nominal uncertain parameter values
        for param, var in original_param_to_vap_map:
            param_to_var_substitution_map_forward[id(param)] = var

        var_to_param_substitution_map_reverse = {}
        # Separation problem initialized to nominal uncertain parameter values
        for var, param in original_var_to_param_map:
            var_to_param_substitution_map_reverse[id(param)] = var

        param_to_var_substitution_map_reverse = {}
        # Separation problem initialized to nominal uncertain parameter values
        for param, var in original_param_to_vap_map:
            param_to_var_substitution_map_reverse[id(var)] = param

        model.swapped_constraints.add(
            replace_expressions(
                expr=replace_expressions(
                    expr=constraint.lower,
                    substitution_map=param_to_var_substitution_map_forward,
                ),
                substitution_map=var_to_param_substitution_map_forward,
            )
            == replace_expressions(
                expr=replace_expressions(
                    expr=constraint.body,
                    substitution_map=param_to_var_substitution_map_forward,
                ),
                substitution_map=var_to_param_substitution_map_forward,
            )
        )

        swapped = model.swapped_constraints[max(model.swapped_constraints.keys())]

        val = generate_standard_repn(swapped.body, compute_values=False)

        if val.constant is not None:
            if type(val.constant) not in native_types:
                temp_expr = replace_expressions(
                    val.constant, substitution_map=var_to_param_substitution_map_reverse
                )
                # We will use generate_standard_repn to generate a
                # simplified expression (in particular, to remove any
                # "0*..." terms)
                temp_expr = generate_standard_repn(temp_expr).to_expression()
                if temp_expr.__class__ not in native_types:
                    model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                elif math.isclose(
                    value(temp_expr),
                    0,
                    rel_tol=COEFF_MATCH_REL_TOL,
                    abs_tol=COEFF_MATCH_ABS_TOL,
                ):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
            elif math.isclose(
                value(val.constant),
                0,
                rel_tol=COEFF_MATCH_REL_TOL,
                abs_tol=COEFF_MATCH_ABS_TOL,
            ):
                pass
            else:
                successful_matching = False
                robust_infeasible = True
        if val.linear_coefs is not None:
            for coeff in val.linear_coefs:
                if type(coeff) not in native_types:
                    temp_expr = replace_expressions(
                        coeff, substitution_map=var_to_param_substitution_map_reverse
                    )
                    # We will use generate_standard_repn to generate a
                    # simplified expression (in particular, to remove any
                    # "0*..." terms)
                    temp_expr = generate_standard_repn(temp_expr).to_expression()
                    if temp_expr.__class__ not in native_types:
                        model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                    elif math.isclose(
                        value(temp_expr),
                        0,
                        rel_tol=COEFF_MATCH_REL_TOL,
                        abs_tol=COEFF_MATCH_ABS_TOL,
                    ):
                        pass
                    else:
                        successful_matching = False
                        robust_infeasible = True
                elif math.isclose(
                    value(coeff),
                    0,
                    rel_tol=COEFF_MATCH_REL_TOL,
                    abs_tol=COEFF_MATCH_ABS_TOL,
                ):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
        if val.quadratic_coefs:
            for coeff in val.quadratic_coefs:
                if type(coeff) not in native_types:
                    temp_expr = replace_expressions(
                        coeff, substitution_map=var_to_param_substitution_map_reverse
                    )
                    # We will use generate_standard_repn to generate a
                    # simplified expression (in particular, to remove any
                    # "0*..." terms)
                    temp_expr = generate_standard_repn(temp_expr).to_expression()
                    if temp_expr.__class__ not in native_types:
                        model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                    elif math.isclose(
                        value(temp_expr),
                        0,
                        rel_tol=COEFF_MATCH_REL_TOL,
                        abs_tol=COEFF_MATCH_ABS_TOL,
                    ):
                        pass
                    else:
                        successful_matching = False
                        robust_infeasible = True
                elif math.isclose(
                    value(coeff),
                    0,
                    rel_tol=COEFF_MATCH_REL_TOL,
                    abs_tol=COEFF_MATCH_ABS_TOL,
                ):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
        if val.nonlinear_expr is not None:
            successful_matching = False
            robust_infeasible = False

        if successful_matching:
            model.util.h_x_q_constraints.add(constraint)

    for i in range(len(list(variables_in_constraint))):
        model.del_component("p_%s" % i)

    for i in range(len(list(params_in_constraint))):
        model.del_component("x_%s" % i)

    model.del_component("swapped_constraints")
    model.del_component("swapped_constraints_index")

    return successful_matching, robust_infeasible


def selective_clone(block, first_stage_vars):
    """
    Clone everything in a base_model except for the first-stage variables
    :param block: the block of the model to be clones
    :param first_stage_vars: the variables which should not be cloned
    :return:
    """
    memo = {'__block_scope__': {id(block): True, id(None): False}}
    for v in first_stage_vars:
        memo[id(v)] = v
    new_block = copy.deepcopy(block, memo)
    new_block._parent = None

    return new_block


def add_decision_rule_variables(model_data, config):
    """
    Add variables for polynomial decision rules to the working
    model.

    Parameters
    ----------
    model_data : ROSolveResults
        Model data.
    config : config_dict
        PyROS solver options.

    Note
    ----
    Decision rule variables are considered first-stage decision
    variables which do not get copied at each iteration.
    PyROS currently supports static (zeroth order),
    affine (first-order), and quadratic DR.
    """
    second_stage_variables = model_data.working_model.util.second_stage_variables
    first_stage_variables = model_data.working_model.util.first_stage_variables
    decision_rule_vars = []

    # since DR expression is a general polynomial in the uncertain
    # parameters, the exact number of DR variables per second-stage
    # variable depends on DR order and uncertainty set dimension
    degree = config.decision_rule_order
    num_uncertain_params = len(model_data.working_model.util.uncertain_params)
    num_dr_vars = sp.special.comb(
        N=num_uncertain_params + degree, k=degree, exact=True, repetition=False
    )

    for idx, ss_var in enumerate(second_stage_variables):
        # declare DR coefficients for current second-stage variable
        indexed_dr_var = Var(
            range(num_dr_vars), initialize=0, bounds=(None, None), domain=Reals
        )
        model_data.working_model.add_component(
            f"decision_rule_var_{idx}", indexed_dr_var
        )

        # index 0 entry of the IndexedVar is the static
        # DR term. initialize to user-provided value of
        # the corresponding second-stage variable.
        # all other entries remain initialized to 0.
        indexed_dr_var[0].set_value(value(ss_var, exception=False))

        # update attributes
        first_stage_variables.extend(indexed_dr_var.values())
        decision_rule_vars.append(indexed_dr_var)

    model_data.working_model.util.decision_rule_vars = decision_rule_vars


def add_decision_rule_constraints(model_data, config):
    """
    Add decision rule equality constraints to the working model.

    Parameters
    ----------
    model_data : ROSolveResults
        Model data.
    config : ConfigDict
        PyROS solver options.
    """

    second_stage_variables = model_data.working_model.util.second_stage_variables
    uncertain_params = model_data.working_model.util.uncertain_params
    decision_rule_eqns = []
    decision_rule_vars_list = model_data.working_model.util.decision_rule_vars
    degree = config.decision_rule_order

    # keeping track of degree of monomial in which each
    # DR coefficient participates will be useful for later
    dr_var_to_exponent_map = ComponentMap()

    # set up uncertain parameter combinations for
    # construction of the monomials of the DR expressions
    monomial_param_combos = []
    for power in range(degree + 1):
        power_combos = it.combinations_with_replacement(uncertain_params, power)
        monomial_param_combos.extend(power_combos)

    # now construct DR equations and declare them on the working model
    second_stage_dr_var_zip = zip(second_stage_variables, decision_rule_vars_list)
    for idx, (ss_var, indexed_dr_var) in enumerate(second_stage_dr_var_zip):
        # for each DR equation, the number of coefficients should match
        # the number of monomial terms exactly
        if len(monomial_param_combos) != len(indexed_dr_var.index_set()):
            raise ValueError(
                f"Mismatch between number of DR coefficient variables "
                f"and number of DR monomials for DR equation index {idx}, "
                f"corresponding to second-stage variable {ss_var.name!r}. "
                f"({len(indexed_dr_var.index_set())}!= {len(monomial_param_combos)})"
            )

        # construct the DR polynomial
        dr_expression = 0
        for dr_var, param_combo in zip(indexed_dr_var.values(), monomial_param_combos):
            dr_expression += dr_var * prod(param_combo)

            # map decision rule var to degree (exponent) of the
            # associated monomial with respect to the uncertain params
            dr_var_to_exponent_map[dr_var] = len(param_combo)

        # declare constraint on model
        dr_eqn = Constraint(expr=dr_expression - ss_var == 0)
        model_data.working_model.add_component(f"decision_rule_eqn_{idx}", dr_eqn)

        # append to list of DR equality constraints
        decision_rule_eqns.append(dr_eqn)

    # finally, add attributes to util block
    model_data.working_model.util.decision_rule_eqns = decision_rule_eqns
    model_data.working_model.util.dr_var_to_exponent_map = dr_var_to_exponent_map


def enforce_dr_degree(blk, config, degree):
    """
    Make decision rule polynomials of a given degree
    by fixing value of the appropriate subset of the decision
    rule coefficients to 0.

    Parameters
    ----------
    blk : ScalarBlock
        Working model, or master problem block.
    config : ConfigDict
        PyROS solver options.
    degree : int
        Degree of the DR polynomials that is to be enforced.
    """
    second_stage_vars = blk.util.second_stage_variables
    indexed_dr_vars = blk.util.decision_rule_vars
    dr_var_to_exponent_map = blk.util.dr_var_to_exponent_map

    for ss_var, indexed_dr_var in zip(second_stage_vars, indexed_dr_vars):
        for dr_var in indexed_dr_var.values():
            dr_var_degree = dr_var_to_exponent_map[dr_var]

            if dr_var_degree > degree:
                dr_var.fix(0)
            else:
                dr_var.unfix()


def identify_objective_functions(model, objective):
    """
    Identify the first and second-stage portions of an Objective
    expression, subject to user-provided variable partitioning and
    uncertain parameter choice. In doing so, the first and second-stage
    objective expressions are added to the model as `Expression`
    attributes.

    Parameters
    ----------
    model : ConcreteModel
        Model of interest.
    objective : Objective
        Objective to be resolved into first and second-stage parts.
    """
    expr_to_split = objective.expr

    has_args = hasattr(expr_to_split, "args")
    is_sum = isinstance(expr_to_split, SumExpression)

    # determine additive terms of the objective expression
    # additive terms are in accordance with user declaration
    if has_args and is_sum:
        obj_args = expr_to_split.args
    else:
        obj_args = [expr_to_split]

    # initialize first and second-stage cost expressions
    first_stage_cost_expr = 0
    second_stage_cost_expr = 0

    first_stage_var_set = ComponentSet(model.util.first_stage_variables)
    uncertain_param_set = ComponentSet(model.util.uncertain_params)

    for term in obj_args:
        non_first_stage_vars_in_term = ComponentSet(
            v for v in identify_variables(term) if v not in first_stage_var_set
        )
        uncertain_params_in_term = ComponentSet(
            param
            for param in identify_mutable_parameters(term)
            if param in uncertain_param_set
        )

        if non_first_stage_vars_in_term or uncertain_params_in_term:
            second_stage_cost_expr += term
        else:
            first_stage_cost_expr += term

    model.first_stage_objective = Expression(expr=first_stage_cost_expr)
    model.second_stage_objective = Expression(expr=second_stage_cost_expr)


def load_final_solution(model_data, master_soln, config):
    '''
    load the final solution into the original model object
    :param model_data: model data container object
    :param master_soln: results data container object returned to user
    :return:
    '''
    if config.objective_focus == ObjectiveType.nominal:
        model = model_data.original_model
        soln = master_soln.nominal_block
    elif config.objective_focus == ObjectiveType.worst_case:
        model = model_data.original_model
        indices = range(len(master_soln.master_model.scenarios))
        k = max(
            indices,
            key=lambda i: value(
                master_soln.master_model.scenarios[i, 0].first_stage_objective
                + master_soln.master_model.scenarios[i, 0].second_stage_objective
            ),
        )
        soln = master_soln.master_model.scenarios[k, 0]

    src_vars = getattr(model, 'tmp_var_list')
    local_vars = getattr(soln, 'tmp_var_list')
    varMap = list(zip(src_vars, local_vars))

    for src, local in varMap:
        src.set_value(local.value, skip_validation=True)

    return


def process_termination_condition_master_problem(config, results):
    '''
    :param config: pyros config
    :param results: solver results object
    :return: tuple (try_backups (True/False)
                  pyros_return_code (default NONE or robust_infeasible or subsolver_error))
    '''
    locally_acceptable = [tc.optimal, tc.locallyOptimal, tc.globallyOptimal]
    globally_acceptable = [tc.optimal, tc.globallyOptimal]
    robust_infeasible = [tc.infeasible]
    try_backups = [
        tc.feasible,
        tc.maxTimeLimit,
        tc.maxIterations,
        tc.maxEvaluations,
        tc.minStepLength,
        tc.minFunctionValue,
        tc.other,
        tc.solverFailure,
        tc.internalSolverError,
        tc.error,
        tc.unbounded,
        tc.infeasibleOrUnbounded,
        tc.invalidProblem,
        tc.intermediateNonInteger,
        tc.noSolution,
        tc.unknown,
    ]

    termination_condition = results.solver.termination_condition
    if config.solve_master_globally == False:
        if termination_condition in locally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, pyrosTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError(
                "This solver return termination condition (%s) "
                "is currently not supported by PyROS." % termination_condition
            )
    else:
        if termination_condition in globally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, pyrosTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError(
                "This solver return termination condition (%s) "
                "is currently not supported by PyROS." % termination_condition
            )


def call_solver(model, solver, config, timing_obj, timer_name, err_msg):
    """
    Solve a model with a given optimizer, keeping track of
    wall time requirements.

    Parameters
    ----------
    model : ConcreteModel
        Model of interest.
    solver : Pyomo solver type
        Subordinate optimizer.
    config : ConfigDict
        PyROS solver settings.
    timing_obj : TimingData
        PyROS solver timing data object.
    timer_name : str
        Name of sub timer under the hierarchical timer contained in
        ``timing_obj`` to start/stop for keeping track of solve
        time requirements.
    err_msg : str
        Message to log through ``config.progress_logger.exception()``
        in event an ApplicationError is raised while attempting to
        solve the model.

    Returns
    -------
    SolverResults
        Solve results. Note that ``results.solver`` contains
        an additional attribute, named after
        ``TIC_TOC_SOLVE_TIME_ATTR``, of which the value is set to the
        recorded solver wall time.

    Raises
    ------
    ApplicationError
        If ApplicationError is raised by the solver.
        In this case, `err_msg` is logged through
        ``config.progress_logger.exception()`` before
        the exception is raised.
    """
    tt_timer = TicTocTimer()

    orig_setting, custom_setting_present = adjust_solver_time_settings(
        timing_obj, solver, config
    )
    timing_obj.start_timer(timer_name)
    tt_timer.tic(msg=None)

    # tentative: reduce risk of InfeasibleConstraintException
    # occurring due to discrepancies between Pyomo NL writer
    # tolerance and (default) subordinate solver (e.g. IPOPT)
    # feasibility tolerances.
    # e.g., a Var fixed outside bounds beyond the Pyomo NL writer
    # tolerance, but still within the default IPOPT feasibility
    # tolerance
    current_nl_writer_tol = pyomo_nl_writer.TOL, pyomo_ampl_repn.TOL
    pyomo_nl_writer.TOL = 1e-4
    pyomo_ampl_repn.TOL = 1e-4

    try:
        results = solver.solve(
            model,
            tee=config.tee,
            load_solutions=False,
            symbolic_solver_labels=config.symbolic_solver_labels,
        )
    except ApplicationError:
        # account for possible external subsolver errors
        # (such as segmentation faults, function evaluation
        # errors, etc.)
        config.progress_logger.error(err_msg)
        raise
    else:
        setattr(
            results.solver, TIC_TOC_SOLVE_TIME_ATTR, tt_timer.toc(msg=None, delta=True)
        )
    finally:
        pyomo_nl_writer.TOL, pyomo_ampl_repn.TOL = current_nl_writer_tol

        timing_obj.stop_timer(timer_name)
        revert_solver_max_time_adjustment(
            solver, orig_setting, custom_setting_present, config
        )

    return results


class IterationLogRecord:
    """
    PyROS solver iteration log record.

    Parameters
    ----------
    iteration : int or None, optional
        Iteration number.
    objective : int or None, optional
        Master problem objective value.
        Note: if the sense of the original model is maximization,
        then this is the negative of the objective value
        of the original model.
    first_stage_var_shift : float or None, optional
        Infinity norm of the difference between first-stage
        variable vectors for the current and previous iterations.
    second_stage_var_shift : float or None, optional
        Infinity norm of the difference between decision rule
        variable vectors for the current and previous iterations.
    dr_polishing_success : bool or None, optional
        True if DR polishing solved successfully, False otherwise.
    num_violated_cons : int or None, optional
        Number of performance constraints found to be violated
        during separation step.
    all_sep_problems_solved : int or None, optional
        True if all separation problems were solved successfully,
        False otherwise (such as if there was a time out, subsolver
        error, or only a subset of the problems were solved due to
        custom constraint prioritization).
    global_separation : bool, optional
        True if separation problems were solved with the subordinate
        global optimizer(s), False otherwise.
    max_violation : int or None
        Maximum scaled violation of any performance constraint
        found during separation step.
    elapsed_time : float, optional
        Total time elapsed up to the current iteration, in seconds.

    Attributes
    ----------
    iteration : int or None
        Iteration number.
    objective : int or None
        Master problem objective value.
        Note: if the sense of the original model is maximization,
        then this is the negative of the objective value
        of the original model.
    first_stage_var_shift : float or None
        Infinity norm of the relative difference between first-stage
        variable vectors for the current and previous iterations.
    second_stage_var_shift : float or None
        Infinity norm of the relative difference between second-stage
        variable vectors (evaluated subject to the nominal uncertain
        parameter realization) for the current and previous iterations.
    dr_var_shift : float or None
        Infinity norm of the relative difference between decision rule
        variable vectors for the current and previous iterations.
        NOTE: This value is not reported in log messages.
    dr_polishing_success : bool or None
        True if DR polishing was solved successfully, False otherwise.
    num_violated_cons : int or None
        Number of performance constraints found to be violated
        during separation step.
    all_sep_problems_solved : int or None
        True if all separation problems were solved successfully,
        False otherwise (such as if there was a time out, subsolver
        error, or only a subset of the problems were solved due to
        custom constraint prioritization).
    global_separation : bool
        True if separation problems were solved with the subordinate
        global optimizer(s), False otherwise.
    max_violation : int or None
        Maximum scaled violation of any performance constraint
        found during separation step.
    elapsed_time : float
        Total time elapsed up to the current iteration, in seconds.
    """

    _LINE_LENGTH = 78
    _ATTR_FORMAT_LENGTHS = {
        "iteration": 5,
        "objective": 13,
        "first_stage_var_shift": 13,
        "second_stage_var_shift": 13,
        "dr_var_shift": 13,
        "num_violated_cons": 8,
        "max_violation": 13,
        "elapsed_time": 13,
    }
    _ATTR_HEADER_NAMES = {
        "iteration": "Itn",
        "objective": "Objective",
        "first_stage_var_shift": "1-Stg Shift",
        "second_stage_var_shift": "2-Stg Shift",
        "dr_var_shift": "DR Shift",
        "num_violated_cons": "#CViol",
        "max_violation": "Max Viol",
        "elapsed_time": "Wall Time (s)",
    }

    def __init__(
        self,
        iteration,
        objective,
        first_stage_var_shift,
        second_stage_var_shift,
        dr_var_shift,
        dr_polishing_success,
        num_violated_cons,
        all_sep_problems_solved,
        global_separation,
        max_violation,
        elapsed_time,
    ):
        """Initialize self (see class docstring)."""
        self.iteration = iteration
        self.objective = objective
        self.first_stage_var_shift = first_stage_var_shift
        self.second_stage_var_shift = second_stage_var_shift
        self.dr_var_shift = dr_var_shift
        self.dr_polishing_success = dr_polishing_success
        self.num_violated_cons = num_violated_cons
        self.all_sep_problems_solved = all_sep_problems_solved
        self.global_separation = global_separation
        self.max_violation = max_violation
        self.elapsed_time = elapsed_time

    def get_log_str(self):
        """Get iteration log string."""
        attrs = [
            "iteration",
            "objective",
            "first_stage_var_shift",
            "second_stage_var_shift",
            # "dr_var_shift",
            "num_violated_cons",
            "max_violation",
            "elapsed_time",
        ]
        return "".join(self._format_record_attr(attr) for attr in attrs)

    def _format_record_attr(self, attr_name):
        """Format attribute record for logging."""
        attr_val = getattr(self, attr_name)
        if attr_val is None:
            fmt_str = f"<{self._ATTR_FORMAT_LENGTHS[attr_name]}s"
            return f"{'-':{fmt_str}}"
        else:
            attr_val_fstrs = {
                "iteration": "f'{attr_val:d}'",
                "objective": "f'{attr_val: .4e}'",
                "first_stage_var_shift": "f'{attr_val:.4e}'",
                "second_stage_var_shift": "f'{attr_val:.4e}'",
                "dr_var_shift": "f'{attr_val:.4e}'",
                "num_violated_cons": "f'{attr_val:d}'",
                "max_violation": "f'{attr_val:.4e}'",
                "elapsed_time": "f'{attr_val:.3f}'",
            }

            # qualifier for DR polishing and separation columns
            if attr_name in ["second_stage_var_shift", "dr_var_shift"]:
                qual = "*" if not self.dr_polishing_success else ""
            elif attr_name == "num_violated_cons":
                qual = "+" if not self.all_sep_problems_solved else ""
            elif attr_name == "max_violation":
                qual = "g" if self.global_separation else ""
            else:
                qual = ""

            attr_val_str = f"{eval(attr_val_fstrs[attr_name])}{qual}"

            return f"{attr_val_str:{f'<{self._ATTR_FORMAT_LENGTHS[attr_name]}'}}"

    def log(self, log_func, **log_func_kwargs):
        """Log self."""
        log_str = self.get_log_str()
        log_func(log_str, **log_func_kwargs)

    @staticmethod
    def get_log_header_str():
        """Get string for iteration log header."""
        fmt_lengths_dict = IterationLogRecord._ATTR_FORMAT_LENGTHS
        header_names_dict = IterationLogRecord._ATTR_HEADER_NAMES
        return "".join(
            f"{header_names_dict[attr]:<{fmt_lengths_dict[attr]}s}"
            for attr in fmt_lengths_dict
            if attr != "dr_var_shift"
        )

    @staticmethod
    def log_header(log_func, with_rules=True, **log_func_kwargs):
        """Log header."""
        if with_rules:
            IterationLogRecord.log_header_rule(log_func, **log_func_kwargs)
        log_func(IterationLogRecord.get_log_header_str(), **log_func_kwargs)
        if with_rules:
            IterationLogRecord.log_header_rule(log_func, **log_func_kwargs)

    @staticmethod
    def log_header_rule(log_func, fillchar="-", **log_func_kwargs):
        """Log header rule."""
        log_func(fillchar * IterationLogRecord._LINE_LENGTH, **log_func_kwargs)
