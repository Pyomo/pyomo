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

from collections import namedtuple
from itertools import chain, count
import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentMap, ComponentSet
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
from pyomo.core.expr import value, EqualityExpression, InequalityExpression
from pyomo.core.expr.numeric_expr import (
    LinearExpression,
    NPV_MaxExpression,
    NPV_MinExpression,
    NPV_SumExpression,
    SumExpression,
)
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.repn.plugins import nl_writer as pyomo_nl_writer
from pyomo.core.expr.visitor import (
    identify_variables,
    identify_mutable_parameters,
    replace_expressions,
)
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
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
PRETRIANGULAR_VAR_COEFF_TOL = 1e-6
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
            f"Found {components_name} "
            "not descended from input model. "
            "Check logger output messages."
        )


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


VariablePartitioning = namedtuple(
    "VariablePartitioning",
    ("first_stage_variables", "second_stage_variables", "state_variables"),
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

    active_model_vars = ComponentSet(
        get_vars_from_components(
            block=model,
            active=True,
            include_fixed=False,
            descend_into=True,
            ctype=(Objective, Constraint),
        )
    )
    check_components_descended_from_model(
        model=model,
        components=active_model_vars,
        components_name=(
            "Vars participating in the "
            "active model Objective/Constraint expressions "
        ),
        config=config,
    )
    check_variables_continuous(model, active_model_vars, config)

    first_stage_vars = (
        ComponentSet(config.first_stage_variables) & active_model_vars
    )
    second_stage_vars = (
        ComponentSet(config.second_stage_variables) & active_model_vars
    )
    state_vars = active_model_vars - (first_stage_vars | second_stage_vars)

    return VariablePartitioning(
        list(first_stage_vars),
        list(second_stage_vars),
        list(state_vars),
    )


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

    Returns
    -------
    user_var_partitioning : VariablePartitioning
        Partitioning of the in-scope model variables into
        first-stage, second-stage, and state variables,
        according to user specification of the first-stage
        and second-stage variables.
    """
    validate_model(model, config)
    user_var_partitioning = validate_variable_partitioning(model, config)
    validate_uncertainty_specification(model, config)
    validate_separation_problem_options(model, config)

    return user_var_partitioning


def get_var_bound_pairs(var):
    """
    Get the domain and declared lower/upper
    bound pairs of a variable data object.

    Parameters
    ----------
    var : VarData
        Variable data object of interest.

    Returns
    -------
    domain_bounds : 2-tuple of None or numeric type
        Domain (lower, upper) bound pair.
    declared_bounds : 2-tuple of None, numeric type, or NumericExpression
        Declared (lower, upper) bound pair.
        Bounds of type `NumericExpression`
        are either constant or mutable expressions.
    """
    # temporarily set domain to Reals to cleanly retrieve
    # the declared bound expressions
    orig_var_domain = var.domain
    var.domain = Reals

    domain_bounds = orig_var_domain.bounds()
    declared_bounds = var.lower, var.upper

    # ensure state of variable object is ultimately left unchanged
    var.domain = orig_var_domain

    return domain_bounds, declared_bounds


def determine_certain_and_uncertain_bound(
        domain_bound,
        declared_bound,
        uncertain_params,
        bound_type,
        ):
    """
    Determine the certain and uncertain lower or upper
    bound for a variable object, based on the specified
    domain and declared bound.

    Parameters
    ----------
    domain_bound : numeric type, NumericExpression, or None
        Domain bound.
    declared_bound : numeric type, NumericExpression, or None
        Declared bound.
    uncertain_params : iterable of ParamData
        Uncertain model parameters.
    bound_type : {'lower', 'upper'}
        Indication of whether the domain bound and declared bound
        specify lower or upper bounds for the variable value.

    Returns
    -------
    certain_bound : numeric type, NumericExpression, or None
        Bound that independent of the uncertain parameters.
    uncertain_bound : numeric expression or None
        Bound that is dependent on the uncertain parameters.
    """
    if bound_type not in {"lower", "upper"}:
        raise ValueError(
            f"Argument {bound_type=!r} should be either 'lower' or 'upper'."
        )

    if declared_bound is not None:
        uncertain_params_in_declared_bound = (
            ComponentSet(uncertain_params)
            & ComponentSet(identify_mutable_parameters(declared_bound))
        )
    else:
        uncertain_params_in_declared_bound = False

    if not uncertain_params_in_declared_bound:
        uncertain_bound = None

        if declared_bound is None:
            certain_bound = domain_bound
        elif domain_bound is None:
            certain_bound = declared_bound
        else:
            if bound_type == "lower":
                certain_bound = (
                    declared_bound if value(declared_bound) >= domain_bound
                    else domain_bound
                )
            else:
                certain_bound = (
                    declared_bound if value(declared_bound) <= domain_bound
                    else domain_bound
                )
    else:
        uncertain_bound = declared_bound
        certain_bound = domain_bound

    return certain_bound, uncertain_bound


BoundTriple = namedtuple(
    "BoundTriple",
    ("lower", "eq", "upper"),
)


def rearrange_bound_pair_to_triple(lower_bound, upper_bound):
    """
    Rearrange a lower/upper bound pair into a lower/equality/upper
    bound triple, according to whether or not the lower and upper
    bound are identical numerical values or expressions.

    Parameters
    ----------
    lower_bound : numeric type, NumericExpression, or None
        Lower bound.
    upper_bound : numeric type, NumericExpression, or None
        Upper bound.

    Returns
    -------
    BoundTriple
        Lower/equality/upper bound triple. The equality
        bound is None if `lower_bound` and `upper_bound`
        are not identical numeric type or ``NumericExpression``
        objects, or else it is set to `upper_bound`,
        in which case, both the lower and upper bounds are
        returned as None.

    Note
    ----
    This method is meant to behave in a manner akin to that of
    ConstraintData.equality, in which a ranged inequality
    constraint may be considered an equality constraint if
    the `lower` and `upper` attributes of the constraint
    are identical and not None.
    """
    if lower_bound is not None and lower_bound is upper_bound:
        eq_bound = upper_bound
        lower_bound = None
        upper_bound = None
    else:
        eq_bound = None

    return BoundTriple(lower_bound, eq_bound, upper_bound)


def get_var_certain_uncertain_bounds(var, uncertain_params):
    """
    Determine the certain and uncertain lower/equality/upper bound
    triples for a variable data object, based on that variable's
    domain and declared bounds.

    Parameters
    ----------
    var : VarData
        Variable data object of interest.
    uncertain_params : iterable of ParamData
        Uncertain model parameters.

    Returns
    -------
    certain_bounds : BoundTriple
        The certain lower/equality/upper bound triple.
    uncertain_bounds : BoundTriple
        The uncertain lower/equality/upper bound triple.
    """
    (domain_lb, domain_ub), (declared_lb, declared_ub) = get_var_bound_pairs(var)

    certain_lb, uncertain_lb = determine_certain_and_uncertain_bound(
        domain_bound=domain_lb,
        declared_bound=declared_lb,
        uncertain_params=uncertain_params,
        bound_type="lower",
    )
    certain_ub, uncertain_ub = determine_certain_and_uncertain_bound(
        domain_bound=domain_ub,
        declared_bound=declared_ub,
        uncertain_params=uncertain_params,
        bound_type="upper",
    )

    certain_bounds = rearrange_bound_pair_to_triple(
        lower_bound=certain_lb, upper_bound=certain_ub,
    )
    uncertain_bounds = rearrange_bound_pair_to_triple(
        lower_bound=uncertain_lb, upper_bound=uncertain_ub
    )

    return certain_bounds, uncertain_bounds


def get_effective_var_partitioning(model_data, config):
    """
    Partition the in-scope variables of the input model
    according to known nonadjustability to the uncertain parameters.
    The result is referred to as the "effective" variable
    partitioning.

    In addition to the first-stage variables,
    some of the variables considered second-stage variables
    or state variables according to the user-provided variable
    partitioning may be nonadjustable. This method analyzes
    the decision rule order, fixed variables, and,
    through an iterative pretriangularization method,
    the equality constraints, to identify nonadjustable variables.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    effective_partitioning : VariablePartitioning
        Effective variable partitioning.
    """
    working_model = model_data.working_model
    user_var_partitioning = model_data.working_model.user_var_partitioning

    # truly nonadjustable variables
    nonadjustable_var_set = ComponentSet()

    # the following variables are immediately known to be nonadjustable:
    # - first-stage variables
    # - (if decision rule order is 0) second-stage variables
    # - all variables fixed to a constant (independent of the uncertain
    #   parameters) explicitly by user or implicitly by bounds
    var_type_list_pairs = (
        ("first-stage", user_var_partitioning.first_stage_variables),
        ("second-stage", user_var_partitioning.second_stage_variables),
        ("state", user_var_partitioning.state_variables),
    )
    for vartype, varlist in var_type_list_pairs:
        for wvar in varlist:
            certain_var_bounds, _ = get_var_certain_uncertain_bounds(
                wvar,
                working_model.uncertain_params,
            )

            is_var_nonadjustable = (
                vartype == "first-stage"
                or (config.decision_rule_order == 0 and vartype == "second-stage")
                or wvar.fixed
                or certain_var_bounds.eq is not None
            )
            if is_var_nonadjustable:
                nonadjustable_var_set.add(wvar)
                config.progress_logger.debug(
                    f"The {vartype} variable {wvar.name!r} "
                    "is nonadjustable, for the following reason(s):"
                )

            if vartype == "first-stage":
                config.progress_logger.debug(f" the variable has a {vartype} status")

            if config.decision_rule_order == 0 and vartype == "second-stage":
                config.progress_logger.debug(
                    f" the variable is {vartype} and the decision rules are static "
                )

            if wvar.fixed:
                config.progress_logger.debug(
                    " the variable is fixed explicitly"
                )

            if certain_var_bounds.eq is not None:
                config.progress_logger.debug(
                    " the variable is fixed by domain/bounds"
                )

    uncertain_params_set = ComponentSet(working_model.uncertain_params)

    # determine constraints that are potentially applicable for
    # pretriangularization
    certain_eq_cons = ComponentSet()
    for wcon in working_model.component_data_objects(Constraint, active=True):
        if not wcon.equality:
            continue
        if ComponentSet(identify_mutable_parameters(wcon.expr)) & uncertain_params_set:
            continue
        certain_eq_cons.add(wcon)

    pretriangular_con_var_map = ComponentMap()
    for num_passes in count(1):
        config.progress_logger.debug(
            f"Performing pass number {num_passes} over the certain constraints."
        )
        new_pretriangular_con_var_map = ComponentMap()
        for ccon in certain_eq_cons:
            vars_in_con = ComponentSet(identify_variables(ccon.body - ccon.upper))
            adj_vars_in_con = vars_in_con - nonadjustable_var_set

            # conditions for pretriangularization of constraint
            # with no uncertain params:
            # - only one nonadjustable variable in the constraint
            # - the nonadjustable variable appears only linearly,
            #   and the linear coefficient exceeds our specified
            #   tolerance.
            if len(adj_vars_in_con) == 1:
                adj_var_in_con = next(iter(adj_vars_in_con))
                ccon_expr_repn = generate_standard_repn(
                    expr=ccon.body - ccon.upper,
                    quadratic=False,
                    compute_values=True,
                )
                adj_var_appears_linearly = (
                    adj_var_in_con not in ComponentSet(ccon_expr_repn.nonlinear_vars)
                    and adj_var_in_con in ComponentSet(ccon_expr_repn.linear_vars)
                )
                if adj_var_appears_linearly:
                    # get coefficient by summation just in case
                    # standard repn does not simplify completely
                    var_linear_coeff = sum(
                        lcoeff
                        for lvar, lcoeff
                        in zip(ccon_expr_repn.linear_vars, ccon_expr_repn.linear_coefs)
                        if lvar is adj_var_in_con
                    )
                    if abs(var_linear_coeff) > PRETRIANGULAR_VAR_COEFF_TOL:
                        new_pretriangular_con_var_map[ccon] = adj_var_in_con
                        config.progress_logger.debug(
                            f" The variable {adj_var_in_con.name!r} is "
                            "made nonadjustable by the pretriangular constraint "
                            f"{ccon.name!r}."
                        )

        nonadjustable_var_set.update(new_pretriangular_con_var_map.values())
        pretriangular_con_var_map.update(new_pretriangular_con_var_map)
        if not new_pretriangular_con_var_map:
            config.progress_logger.debug(
                "No new pretriangular constraint/variable pairs found. "
                "Terminating pretriangularization loop."
            )
            break

        for pcon in new_pretriangular_con_var_map:
            certain_eq_cons.remove(pcon)

    pretriangular_vars = ComponentSet(pretriangular_con_var_map.values())
    config.progress_logger.debug(
        f"Identified {len(pretriangular_con_var_map)} pretriangular "
        f"constraints and {len(pretriangular_vars)} pretriangular variables "
        f"in {num_passes} passes over the certain constraints."
    )

    effective_first_stage_vars = list(nonadjustable_var_set)
    effective_second_stage_vars = [
        var
        for var in user_var_partitioning.second_stage_variables
        if var not in nonadjustable_var_set
    ]
    effective_state_vars = [
        var
        for var in user_var_partitioning.state_variables
        if var not in nonadjustable_var_set
    ]
    num_vars = len(
        effective_first_stage_vars
        + effective_second_stage_vars
        + effective_state_vars
    )

    config.progress_logger.debug("Effective partitioning statistics:")
    config.progress_logger.debug(
        f"  Variables: {num_vars}"
    )
    config.progress_logger.debug(
        f"    Effective first-stage variables: {len(effective_first_stage_vars)}"
    )
    config.progress_logger.debug(
        f"    Effective second-stage variables: {len(effective_second_stage_vars)}"
    )
    config.progress_logger.debug(
        f"    Effective state variables: {len(effective_state_vars)}"
    )

    return VariablePartitioning(
        first_stage_variables=effective_first_stage_vars,
        second_stage_variables=effective_second_stage_vars,
        state_variables=effective_state_vars,
    )


def add_effective_var_partitioning(model_data, config):
    """
    Obtain a repartitioning of the in-scope variables of the
    working model according to known adjustability to the
    uncertain parameters, and add this repartitioning to the
    working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver options.
    """
    effective_partitioning = get_effective_var_partitioning(
        model_data=model_data,
        config=config,
    )
    model_data.working_model.effective_var_partitioning = (
        VariablePartitioning(**effective_partitioning._asdict())
    )


def create_bound_constraint_expr(expr, bound, bound_type):
    """
    Create a relational expression establishing a bound
    for a numeric expression of interest.

    The expression is such that the bound appears on the
    right-hand side of the relational (inequality/equality)
    operator.

    Parameters
    ----------
    expr : NumericValue
        Expression for which a bound is to be imposed.
        This can be a Pyomo expression, Var, or Param.
    bound : native numeric type or NumericValue
        Bound for `expr`. This should be a numeric constant,
        Param, or constant/mutable Pyomo expression.
    bound_type : {'lower', 'eq', 'upper'}
        Indicator for whether `expr` is to be lower bounded,
        equality bounded, or upper bounded, by `bound`.

    Returns
    -------
    RelationalExpression
        Establishes a bound on `expr`.
    """
    if bound_type == "lower":
        return -expr <= -bound
    elif bound_type == "eq":
        return expr == bound
    elif bound_type == "upper":
        return expr <= bound
    else:
        raise ValueError(
            f"Bound type {bound_type!r} not supported."
        )


def remove_var_declared_bound(var, bound_type):
    """
    Remove the specified declared bound(s) of a variable data object.

    Parameters
    ----------
    var : VarData
        Variable data object of interest.
    bound_type : {'lower', 'eq', 'upper'}
        Indicator for the declared bound(s) to remove.
        Note: if 'eq' is specified, then both the
        lower and upper bounds are removed.
    """
    if bound_type == "lower":
        var.setlb(None)
    elif bound_type == "eq":
        var.setlb(None)
        var.setub(None)
    elif bound_type == "upper":
        var.setub(None)
    else:
        raise ValueError(
            f"Bound type {bound_type!r} not supported. "
            "Bound type must be 'lower', 'eq, or 'upper'."
        )


def remove_all_var_bounds(var):
    """
    Remove all the domain and declared bounds for a specified
    variable data object.
    """
    var.setlb(None)
    var.setub(None)
    var.domain = Reals


def turn_nonadjustable_var_bounds_to_constraints(model_data):
    """
    Reformulate uncertain bounds for the nonadjustable
    (i.e. effective first-stage) variables of the working
    model to constraints.

    Only uncertain declared bounds are reformulated to
    constraints, as these are the only bounds we need to
    reformulate to properly construct the subproblems.
    Consequently, all constraints added to the working model
    in this method are considered performance constraints.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver settings.
    """
    working_model = model_data.working_model
    performance_eq_cons = working_model.effective_performance_equality_cons
    performance_ineq_cons = working_model.effective_performance_inequality_cons

    nonadjustable_vars = (
        working_model.effective_var_partitioning.first_stage_variables
    )
    uncertain_params_set = ComponentSet(working_model.uncertain_params)
    for var in nonadjustable_vars:
        _, declared_bounds = get_var_bound_pairs(var)
        declared_bound_triple = rearrange_bound_pair_to_triple(*declared_bounds)
        var_name = var.getname(
            relative_to=working_model.user_model,
            fully_qualified=True,
        )
        for btype, bound in declared_bound_triple._asdict().items():
            is_bound_uncertain = (
                bound is not None
                and (
                    ComponentSet(identify_mutable_parameters(bound))
                    & uncertain_params_set
                )
            )
            if is_bound_uncertain:
                var_bound_con = Constraint(
                    expr=create_bound_constraint_expr(var, bound, btype),
                )
                working_model.user_model.add_component(
                    unique_component_name(
                        working_model.user_model,
                        f"var_{var_name}_uncertain_{btype}_bound_con",
                    ),
                    var_bound_con,
                )
                remove_var_declared_bound(var, btype)

                if btype == "eq":
                    performance_eq_cons.append(var_bound_con)
                else:
                    performance_ineq_cons.append(var_bound_con)

    # for subsequent developments: return a mapping
    # from each variable to the corresponding binding constraints?
    # we will add this as needed when changes are made to
    # the interface for separation priority ordering


def turn_adjustable_var_bounds_to_constraints(model_data):
    """
    Reformulate domain and declared bounds for the
    adjustable (i.e., effective second-stage and effective state)
    variables of the working model to explicit constraints.

    The domain and declared bounds for every adjustable variable
    are unconditionally reformulated to constraints,
    as this is required for appropriate construction of the
    subproblems later.
    Since these constraints depend on adjustable variables,
    they are taken to be (effective) performance constraints.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver settings.
    """
    working_model = model_data.working_model
    performance_eq_cons = working_model.effective_performance_equality_cons
    performance_ineq_cons = working_model.effective_performance_inequality_cons

    adjustable_vars = (
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    for var in adjustable_vars:
        cert_bound_triple, uncert_bound_triple = get_var_certain_uncertain_bounds(
            var, working_model.uncertain_params,
        )
        var_name = var.getname(
            relative_to=working_model.user_model,
            fully_qualified=True,
        )
        cert_uncert_bound_zip = (
            ("certain", cert_bound_triple),
            ("uncertain", uncert_bound_triple),
        )
        for certainty_desc, bound_triple in cert_uncert_bound_zip:
            for btype, bound in bound_triple._asdict().items():
                if bound is not None:
                    var_bound_con = Constraint(
                        expr=create_bound_constraint_expr(var, bound, btype),
                    )
                    working_model.user_model.add_component(
                        unique_component_name(
                            working_model.user_model,
                            f"var_{var_name}_{certainty_desc}_{btype}_bound_con",
                        ),
                        var_bound_con,
                    )
                    if btype == "eq":
                        performance_eq_cons.append(var_bound_con)
                    else:
                        performance_ineq_cons.append(var_bound_con)

        remove_all_var_bounds(var)

    # for subsequent developments: return a mapping
    # from each variable to the corresponding binding constraints?
    # we will add this as needed when changes are made to
    # the interface for separation priority ordering


def setup_working_model(model_data, config, user_var_partitioning):
    """
    Set up (construct) the working model based on user inputs,
    and add it to the model data object.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solve settings.
    user_var_partitioning : VariablePartitioning
        User-based partitioning of the in-scope
        variables of the input model.
    """
    original_model = model_data.original_model

    # add temporary block to help keep track of variables
    # and uncertain parameters after cloning
    temp_util_block_attr_name = unique_component_name(
        original_model, "util"
    )
    original_model.add_component(temp_util_block_attr_name, Block())
    orig_temp_util_block = getattr(original_model, temp_util_block_attr_name)
    orig_temp_util_block.uncertain_params = config.uncertain_params
    orig_temp_util_block.user_var_partitioning = VariablePartitioning(
        **user_var_partitioning._asdict()
    )

    # now set up working model
    model_data.working_model = working_model = ConcreteModel()

    # original user model will be a sub-block of working model,
    # in order to avoid attribute name clashes later
    working_model.user_model = original_model.clone()

    # facilitate later retrieval of the user var partitioning
    working_temp_util_block = getattr(
        working_model.user_model,
        temp_util_block_attr_name,
    )
    model_data.working_model.uncertain_params = (
        working_temp_util_block.uncertain_params.copy()
    )
    working_model.user_var_partitioning = VariablePartitioning(
        **working_temp_util_block.user_var_partitioning._asdict()
    )

    # we are done with the util blocks
    delattr(original_model, temp_util_block_attr_name)
    delattr(working_model.user_model, temp_util_block_attr_name)

    # keep track of the original active constraints
    working_model.original_active_equality_cons = []
    working_model.original_active_inequality_cons = []
    for con in working_model.component_data_objects(Constraint, active=True):
        if con.equality:
            # note: ranged constraints with identical LHS and RHS
            #       objects are considered equality constraints
            working_model.original_active_equality_cons.append(con)
        else:
            working_model.original_active_inequality_cons.append(con)

    # partition the constraints according to their
    # status as equality/inequality constraints
    # and their dependence on adjustable variables or uncertain
    # parameters.
    # we will need this later for construction of the subproblems
    working_model.effective_first_stage_equality_cons = []
    working_model.effective_first_stage_inequality_cons = []
    working_model.effective_performance_equality_cons = []
    working_model.effective_performance_inequality_cons = []


def remove_con_declared_bound(con, bound_type):
    """
    Remove a bound in a constraint data expression.
    """
    if bound_type == "lower":
        con.set_value((None, con.body, con.upper))
    elif bound_type == "eq":
        con.set_value((None, con.body, None))
    elif bound_type == "upper":
        con.set_value((con.lower, con.body, None))
    else:
        raise ValueError(
            f"Bound type {bound_type} not supported."
        )


def standardize_inequality_constraints(model_data):
    """
    Standardize the inequality constraints of the working model,
    and classify them as first-stage inequalities or performance
    (i.e., second-stage) inequalities.

    Parameters
    ----------
    model_data : model data object
        Main model data object, containing the working model.
    """
    working_model = model_data.working_model
    uncertain_params_set = ComponentSet(working_model.uncertain_params)
    adjustable_vars_set = ComponentSet(
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    for con in working_model.original_active_inequality_cons:
        uncertain_params_in_con_expr = (
            ComponentSet(identify_mutable_parameters(con.expr))
            & uncertain_params_set
        )
        adjustable_vars_in_con_body = (
            ComponentSet(identify_variables(con.body))
            & adjustable_vars_set
        )

        if uncertain_params_in_con_expr | adjustable_vars_in_con_body:
            con_bounds_triple = rearrange_bound_pair_to_triple(
                lower_bound=con.lower,
                upper_bound=con.upper,
            )
            finite_bounds = {
                btype: bd for btype, bd in con_bounds_triple._asdict().items()
                if bd is not None
            }
            for btype, bound in finite_bounds.items():
                if btype == "eq":
                    # no equality bounds should be identified here.
                    # equality bound may be identified if:
                    # 1. bound rearrangement method has a bug
                    # 2. ConstraintData.equality is changed.
                    #    such a change would affect this method
                    #    only indirectly
                    raise ValueError(
                        f"Found an equality bound {bound} for the constraint "
                        f"for the constraint with name {con.name!r}. "
                        "Either the bound or the constraint has been misclassified."
                        "Report this case to the Pyomo/PyROS developers."
                    )

                uncertain_params_in_bound = ComponentSet(
                    identify_mutable_parameters(bound)
                ) & uncertain_params_set

                if adjustable_vars_in_con_body | uncertain_params_in_bound:
                    if len(finite_bounds) == 1:
                        # modify constraints with only a single inequality
                        # operator in place, for efficiency
                        con.set_value(
                            create_bound_constraint_expr(con.body, bound, btype)
                        )
                        new_con = con
                    else:
                        # ranged constraint: declare a new constraint for
                        # each of the performance inequalities;
                        # first-stage inequalities remain in place
                        new_con_name = con.getname(
                            relative_to=working_model.user_model,
                            fully_qualified=True,
                        )
                        new_con = Constraint(
                            expr=create_bound_constraint_expr(con.body, bound, btype)
                        )
                        working_model.user_model.add_component(
                            f"con_{new_con_name}_{btype}_bound_con",
                            new_con,
                        )
                        remove_con_declared_bound(con, btype)
                    working_model.effective_performance_inequality_cons.append(new_con)
                else:
                    # constraint has a first-stage inequality (bound)
                    # this inequality (bound) will not be modified
                    working_model.effective_first_stage_inequality_cons.append(con)

            if con.lower is None and con.upper is None:
                # either the original constraint had no bounds,
                # or the inequalities (bounds) have been stripped
                # and used to declare performance constraints
                con.deactivate()
        else:
            # constraint depends on the nonadjustable variables only
            working_model.effective_first_stage_inequality_cons.append(con)

    # for subsequent developments: map the original constraints
    # to the derived performance inequalities?
    # we will add this as needed when changes are made to
    # the interface for separation priority ordering


def standardize_equality_constraints(model_data):
    """
    Classify the original active equality constraints of the
    working model as first-stage or performance constraints.

    Parameters
    ----------
    model_data : model data object
        Main model data object, containing the working model.
    """
    working_model = model_data.working_model
    uncertain_params_set = ComponentSet(working_model.uncertain_params)
    adjustable_vars_set = ComponentSet(
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    for con in working_model.original_active_equality_cons:
        uncertain_params_in_con_expr = (
            ComponentSet(identify_mutable_parameters(con.expr))
            & uncertain_params_set
        )
        adjustable_vars_in_con_body = (
            ComponentSet(identify_variables(con.body))
            & adjustable_vars_set
        )

        # note: none of the equality constraint expressions are modified
        if uncertain_params_in_con_expr | adjustable_vars_in_con_body:
            working_model.effective_performance_equality_cons.append(con)
        else:
            working_model.effective_first_stage_equality_cons.append(con)


def get_summands(expr):
    """
    Recursively gather the individual summands of a numeric expression.

    Parameters
    ----------
    expr : native numeric type or NumericValue
        Expression to be analyzed.

    Returns
    -------
    summands : list of expression-like
        The summands.
    """
    if isinstance(expr, SumExpression):
        # note: NPV_SumExpression and LinearExpression
        #       are subclasses of SumExpression,
        #       so those instances are decomposed here, as well.
        summands = []
        for arg in expr.args:
            summands.extend(get_summands(arg))
    else:
        summands = [expr]
    return summands


def declare_objective_expressions(working_model, objective, sense=minimize):
    """
    Identify the per-stage summands of an objective of interest,
    according to the user-based variable partitioning.

    Two Expressions are declared on the working model to contain
    the per-stage summands:

    - ``first_stage_objective``: Sum of additive terms of `objective`
      that are non-uncertain constants or depend only on the
      user-defined first-stage variables.
    - ``second_stage_objective``: Sum of all other additive terms of
      `objective`.

    To facilitate retrieval of the original objective expression
    (modified to account for the sense), an Expression called
    ``full_objective`` is also declared on the working model.

    Parameters
    ----------
    working_model : ConcreteModel
        Working model, constructed during a PyROS solver run.
    objective : ObjectiveData
        Objective of which summands are to be identified.
    sense : {common.enums.minimize, common.enums.maximize}, optional
        Desired sense of the objective; default is minimize.
    """
    if sense not in {minimize, maximize}:
        raise ValueError(
            f"Objective sense {sense} not supported. "
            f"Ensure sense is {minimize} (minimize) or {maximize} (maximize)."
        )

    obj_expr = objective.expr

    obj_args = get_summands(obj_expr)

    # initialize first and second-stage cost expressions
    first_stage_expr = 0
    second_stage_expr = 0

    first_stage_var_set = ComponentSet(
        working_model.user_var_partitioning.first_stage_variables
    )
    uncertain_param_set = ComponentSet(working_model.uncertain_params)

    obj_sense = objective.sense
    for term in obj_args:
        non_first_stage_vars_in_term = ComponentSet(
            v for v in identify_variables(term) if v not in first_stage_var_set
        )
        uncertain_params_in_term = ComponentSet(
            param
            for param in identify_mutable_parameters(term)
            if param in uncertain_param_set
        )

        # account for objective sense

        # update all expressions
        std_term = term if obj_sense == sense else -term
        if non_first_stage_vars_in_term or uncertain_params_in_term:
            second_stage_expr += std_term
        else:
            first_stage_expr += std_term

    working_model.first_stage_objective = Expression(expr=first_stage_expr)
    working_model.second_stage_objective = Expression(expr=second_stage_expr)

    # useful for later
    working_model.full_objective = Expression(
        expr=obj_expr if sense == obj_sense else -obj_expr
    )


def standardize_active_objective(model_data, config):
    """
    Standardize the active objective of the working model.

    This method involves declaration of:

    - named expressions for the full active objective
      (in a minimization sense), the first-stage objective summand,
      and the second-stage objective summand.
    - an epigraph epigraph variable and constraint.

    The epigraph constraint is considered a first-stage
    inequality provided that it is independent of the
    adjustable (i.e., effective second-stage and effective state)
    variables and the uncertain parameters.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    """
    working_model = model_data.working_model

    active_obj = next(
        working_model.component_data_objects(
            Objective, active=True, descend_into=True
        )
    )
    model_data.active_obj_original_sense = active_obj.sense

    # per-stage summands will be useful for reporting later
    declare_objective_expressions(
        working_model=working_model,
        objective=active_obj,
    )

    # epigraph reformulation components will be useful for later
    working_model.epigraph_var = Var(initialize=value(active_obj, exception=False))
    working_model.epigraph_con = Constraint(
        expr=working_model.full_objective.expr - working_model.epigraph_var <= 0
    )

    # we add the epigraph objective later, as needed,
    # on a per subproblem basis;
    # doing so is more efficient than adding the objective now
    active_obj.deactivate()

    # classify the epigraph constraint
    adjustable_vars = (
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    uncertain_params_in_obj = (
        ComponentSet(identify_mutable_parameters(active_obj.expr))
        & ComponentSet(working_model.uncertain_params)
    )
    adjustable_vars_in_obj = (
        ComponentSet(identify_variables(active_obj.expr))
        & adjustable_vars
    )
    if (uncertain_params_in_obj | adjustable_vars_in_obj):
        if config.objective_focus == ObjectiveType.worst_case:
            working_model.effective_performance_inequality_cons.append(
                working_model.epigraph_con
            )
        elif config.objective_focus == ObjectiveType.nominal:
            working_model.effective_first_stage_inequality_cons.append(
                working_model.epigraph_con
            )
        else:
            raise ValueError(
                "Classification of the epigraph constraint with uncertain "
                "and/or adjustable components not implemented "
                f"for objective focus {config.objective_focus!r}."
            )
    else:
        working_model.effective_first_stage_inequality_cons.append(
            working_model.epigraph_con
        )


def new_add_decision_rule_variables(model_data, config):
    """
    Add variables parameterizing the (polynomial)
    decision rules to the working model.

    Parameters
    ----------
    model_data : model data object
        Model data.
    config : ConfigDict
        PyROS solver options.

    Notes
    -----
    1. One set of decision rule variables is added for each
       effective second-stage variable.
    2. As an efficiency, no decision rule variables
       are added for the nonadjustable, user-defined second-stage
       variables, since the decision rules for such variables
       are necessarily nonstatic.
    """
    effective_second_stage_vars = (
        model_data.working_model.effective_var_partitioning.second_stage_variables
    )
    model_data.working_model.decision_rule_vars = decision_rule_vars = []

    # facilitate matching of effective second-stage vars to DR vars later
    model_data.working_model.eff_ss_var_to_dr_var_map = eff_ss_var_to_dr_var_map = (
        ComponentMap()
    )

    # since DR expression is a general polynomial in the uncertain
    # parameters, the exact number of DR variables
    # per effective second-stage variable
    # depends only on the DR order and uncertainty set dimension
    degree = config.decision_rule_order
    num_uncertain_params = len(model_data.working_model.uncertain_params)
    num_dr_vars = sp.special.comb(
        N=num_uncertain_params + degree, k=degree, exact=True, repetition=False
    )

    for idx, eff_ss_var in enumerate(effective_second_stage_vars):
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
        indexed_dr_var[0].set_value(value(eff_ss_var, exception=False))

        # update attributes
        decision_rule_vars.append(indexed_dr_var)
        eff_ss_var_to_dr_var_map[eff_ss_var] = indexed_dr_var


def new_add_decision_rule_constraints(model_data, config):
    """
    Add decision rule equality constraints to the working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver options.
    """

    effective_second_stage_vars = (
        model_data.working_model.effective_var_partitioning.second_stage_variables
    )
    indexed_dr_var_list = model_data.working_model.decision_rule_vars
    uncertain_params = model_data.working_model.uncertain_params
    degree = config.decision_rule_order

    model_data.working_model.decision_rule_eqns = decision_rule_eqns = []

    # keeping track of degree of monomial
    # (in terms of the uncertain parameters)
    # in which each DR coefficient participates will be useful for
    # later
    model_data.working_model.dr_var_to_exponent_map = dr_var_to_exponent_map = (
        ComponentMap()
    )

    # facilitate retrieval of DR equation for a given
    # effective second-stage variable later
    model_data.working_model.eff_ss_var_to_dr_eqn_map = eff_ss_var_to_dr_eqn_map = (
        ComponentMap()
    )

    # set up uncertain parameter combinations for
    # construction of the monomials of the DR expressions
    monomial_param_combos = []
    for power in range(degree + 1):
        power_combos = it.combinations_with_replacement(uncertain_params, power)
        monomial_param_combos.extend(power_combos)

    # now construct DR equations and declare them on the working model
    second_stage_dr_var_zip = zip(effective_second_stage_vars, indexed_dr_var_list)
    for idx, (eff_ss_var, indexed_dr_var) in enumerate(second_stage_dr_var_zip):
        # for each DR equation, the number of coefficients should match
        # the number of monomial terms exactly
        if len(monomial_param_combos) != len(indexed_dr_var.index_set()):
            raise ValueError(
                f"Mismatch between number of DR coefficient variables "
                f"and number of DR monomials for DR equation index {idx}, "
                "corresponding to effective second-stage variable "
                f"{eff_ss_var.name!r}. "
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
        dr_eqn = Constraint(expr=dr_expression - eff_ss_var == 0)
        model_data.working_model.add_component(f"decision_rule_eqn_{idx}", dr_eqn)

        decision_rule_eqns.append(dr_eqn)
        eff_ss_var_to_dr_eqn_map[eff_ss_var] = dr_eqn


def get_all_nonadjustable_variables(working_model):
    """
    Get all nonadjustable variables of the working model.

    The nonadjustable variables comprise the:

    - epigraph variable
    - decision rule variables
    - effective first-stage variables
    """
    epigraph_var = working_model.epigraph_var
    decision_rule_vars = list(
        generate_all_decision_rule_var_data_objects(working_model)
    )
    effective_first_stage_vars = (
        working_model.effective_var_partitioning.first_stage_variables
    )

    return [epigraph_var] + decision_rule_vars + effective_first_stage_vars


def generate_all_decision_rule_var_data_objects(working_blk):
    """
    Generate a sequence of all decision rule variable data
    objects.

    Parameters
    ----------
    working_blk : BlockData
        Block with a structure similar to the working model
        created during preprocessing.

    Yields
    ------
    VarData
        Decision rule variable.
    """
    for indexed_var in working_blk.decision_rule_vars:
        yield from indexed_var.values()


def generate_all_decision_rule_eqns(working_blk):
    """
    Generate sequence of all decision rule equations.
    """
    for indexed_con in working_blk.decision_rule_eqns:
        yield from indexed_con.values()


def get_dr_expression(working_blk, second_stage_var):
    """
    Get DR expression corresponding to given second-stage variable.

    Parameters
    ----------
    working_blk : BlockData
        Block with a structure similar to the working model
        created during preprocessing.

    Returns
    ------
    VarData, LinearExpression, or SumExpression
        The corresponding DR expression.
    """
    dr_con = working_blk.eff_ss_var_to_dr_eqn_map[second_stage_var]
    return sum(dr_con.body.args[:-1])


def perform_coefficient_matching(model_data, config):
    """
    Perform coefficient matching reformulation
    of some performance equality constraints.

    Every performance equality constraint that is independent
    of the state variables can potentially be simplified to a
    set of first-stage equality constraints.

    In some cases, robust infeasibility can be detected.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    robust_infeasible : bool
        True if model found to be robust infeasible,
        False otherwise.
    """
    working_model = model_data.working_model
    ep = working_model.effective_var_partitioning

    effective_second_stage_var_set = ComponentSet(ep.second_stage_variables)
    effective_state_var_set = ComponentSet(ep.state_variables)
    all_vars_set = ComponentSet(working_model.all_variables)
    originally_unfixed_vars = [var for var in all_vars_set if not var.fixed]

    # we will need this to substitute DR expressions for
    # second-stage variables later
    ssvar_id_to_dr_expr_map = {
        id(ss_var): get_dr_expression(working_model, ss_var)
        for ss_var in effective_second_stage_var_set
    }

    # goal: examine constraint expressions in terms of the
    #       uncertain params. we will use standard repn to do this.
    # standard repn analyzes expressions in terms of Var components,
    # but the uncertain params are implemented as mutable Param objects
    # so we temporarily define Var components to be briefly substituted
    # for the uncertain parameters as the constraints are analyzed
    uncertain_params_set = ComponentSet(working_model.uncertain_params)
    working_model.temp_param_vars = temp_param_vars = Var(
        range(len(uncertain_params_set)),
        initialize={
            idx: value(param) for idx, param in enumerate(uncertain_params_set)
        }
    )
    uncertain_param_to_temp_var_map = ComponentMap(
        (param, param_var)
        for param, param_var
        in zip(uncertain_params_set, temp_param_vars.values())
    )
    uncertain_param_id_to_temp_var_map = {
        id(param): var for param, var in uncertain_param_to_temp_var_map.items()
    }

    # constraints generated during the reformulation will be placed here
    working_model.coefficient_matching_conlist = coeff_matching_conlist = (
        ConstraintList()
    )

    performance_eq_cons = working_model.effective_performance_equality_cons.copy()
    for con in performance_eq_cons:
        vars_in_con = ComponentSet(identify_variables(con.expr))
        mutable_params_in_con = ComponentSet(identify_mutable_parameters(con.expr))

        second_stage_vars_in_con = vars_in_con & effective_second_stage_var_set
        state_vars_in_con = vars_in_con & effective_state_var_set
        uncertain_params_in_con = mutable_params_in_con & uncertain_params_set

        coefficient_matching_applicable = (
            not state_vars_in_con
            and (uncertain_params_in_con or second_stage_vars_in_con)
        )
        if coefficient_matching_applicable:
            con_expr_after_dr_substitution = replace_expressions(
                expr=con.body - con.upper,
                substitution_map=ssvar_id_to_dr_expr_map,
            )

            # substitute temporarily defined vars for uncertain params.
            # note: this is performed after, rather than along with,
            # the DR expression substitution, as the DR expressions
            # contain uncertain params
            con_expr_after_all_substitutions = replace_expressions(
                expr=con_expr_after_dr_substitution,
                substitution_map=uncertain_param_id_to_temp_var_map,
            )

            # analyze the expression with respect to the
            # uncertain parameters only. thus, only the proxy
            # variables for the uncertain parameters are unfixed
            # during the analysis
            for var in originally_unfixed_vars:
                var.fix()
            expr_repn = generate_standard_repn(
                expr=con_expr_after_all_substitutions,
                compute_values=False,
            )

            # ensure state of every variable remains unchanged
            # when done
            for var in originally_unfixed_vars:
                var.unfix()

            if expr_repn.nonlinear_expr is not None:
                config.progress_logger.debug(
                    f"Equality constraint {con.name!r} "
                    "is state-variable independent, but cannot be written "
                    "as a polynomial in the uncertain parameters with "
                    "the currently available expression analyzers "
                    "and selected decision rules "
                    f"(decision_rule_order={config.decision_rule_order}). "
                    "We are unable to write a coefficent matching reformulation "
                    "of this constraint."
                )

                # nothing we can do to reformulate this constraint,
                # so move on
                continue

            polynomial_repn_coeffs = (
                [expr_repn.constant]
                + list(expr_repn.linear_coefs)
                + list(expr_repn.quadratic_coefs)
            )
            for coef_expr in polynomial_repn_coeffs:
                simplified_coef_expr = generate_standard_repn(
                    expr=coef_expr,
                    compute_values=True,
                ).to_expression()

                # for robust satisfaction of the original equality
                # constraint, all polynomial coefficients must be
                # equal to zero. so for each coefficient,
                # we either check for trivial robust
                # feasibility/infeasibility, or add a constraint
                # restricting the coefficient expression to value 0
                if isinstance(simplified_coef_expr, tuple(native_types)):
                    # coefficient is a constant;
                    # check value to determine
                    # trivial feasibility/infeasibility
                    robust_infeasible = not math.isclose(
                        a=simplified_coef_expr,
                        b=0,
                        rel_tol=COEFF_MATCH_REL_TOL,
                        abs_tol=COEFF_MATCH_ABS_TOL,
                    )
                    if robust_infeasible:
                        config.progress_logger.info(
                            "PyROS has determined that the model is "
                            "robust infeasible. "
                            "One reason for this is that "
                            f"the equality constraint {con.name!r} "
                            "cannot be satisfied against all realizations "
                            "of uncertainty, "
                            "given the current partitioning into "
                            "first-stage, second-stage, and state variables. "
                            "Consider editing this constraint to reference some "
                            "(additional) second-stage and/or state variable(s)."
                        )

                        # robust infeasibility found;
                        # that is sufficient for termination of PyROS.
                        return robust_infeasible

                else:
                    # coefficient is dependent on model first-stage
                    # and DR variables. add matching constraint
                    coeff_matching_conlist.add(simplified_coef_expr == 0)

                    # matching constraint depends on nonadjustable
                    # variables only, so it is first-stage
                    last_idx = coeff_matching_conlist.index_set().last()
                    working_model.effective_first_stage_equality_cons.append(
                        coeff_matching_conlist[last_idx]
                    )

                    config.progress_logger.debug(
                        f"Derived from constraint {con.name!r} a coefficient "
                        "matching constraint with expression: \n    "
                        f"{coeff_matching_conlist[last_idx].expr}."
                    )

            # constraint has been reformulated out of the model,
            # i.e., coefficients have all been matched or found
            #       to yield trivial satisfaction of the constraint
            con.deactivate()
            working_model.effective_performance_equality_cons.remove(con)

    # we no longer need these auxiliary components
    working_model.del_component(temp_param_vars)
    working_model.del_component(temp_param_vars.index_set())

    return False


def new_preprocess_model_data(model_data, config, user_var_partitioning):
    """
    Preprocess user inputs to modeling objects from which
    PyROS subproblems can be efficiently constructed.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver options.
    user_var_partitioning : VariablePartitioning
        User-based partitioning of the in-scope
        variables of the input model.

    Returns
    -------
    robust_infeasible : bool
        True if RO problem was found to be robust infeasible,
        False otherwise.
    """
    setup_working_model(model_data, config, user_var_partitioning)

    # extract as many truly nonadjustable variables as possible
    # from the second-stage and state variables
    config.progress_logger.debug("Repartitioning variables by nonadjustability...")
    add_effective_var_partitioning(model_data, config)

    # different treatment for effective first-stage
    # than for effective second-stage and state variables
    config.progress_logger.debug("Turning some variable bounds to constraints...")
    turn_nonadjustable_var_bounds_to_constraints(model_data)
    turn_adjustable_var_bounds_to_constraints(model_data)

    config.progress_logger.debug("Standardizing the model constraints...")
    standardize_inequality_constraints(model_data)
    standardize_equality_constraints(model_data)

    # includes epigraph reformulation
    config.progress_logger.debug("Standardizing the active objective...")
    standardize_active_objective(model_data, config)

    # DR components are added only per effective second-stage variable
    config.progress_logger.debug("Adding decision rule components...")
    new_add_decision_rule_variables(model_data, config)
    new_add_decision_rule_constraints(model_data, config)

    # the epigraph and DR variables are also first-stage
    config.progress_logger.debug("Finalizing nonadjustable variables...")
    model_data.working_model.all_nonadjustable_variables = (
        get_all_nonadjustable_variables(model_data.working_model)
    )
    model_data.working_model.all_variables = (
        model_data.working_model.all_nonadjustable_variables
        + model_data.working_model.effective_var_partitioning.second_stage_variables
        + model_data.working_model.effective_var_partitioning.state_variables
    )

    config.progress_logger.debug("Performing coefficient matching reformulation...")
    robust_infeasible = perform_coefficient_matching(model_data, config)

    return robust_infeasible


def log_model_statistics(model_data, config):
    """
    Log statistics for the preprocessed model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    config : ConfigDict
        PyROS solver settings.
    """
    working_model = model_data.working_model

    ep = working_model.effective_var_partitioning
    up = working_model.user_var_partitioning

    # variables. we log the user partitioning
    num_vars = len(working_model.all_variables)
    num_epigraph_vars = 1
    num_first_stage_vars = len(up.first_stage_variables)
    num_second_stage_vars = len(up.second_stage_variables)
    num_state_vars = len(up.state_variables)
    num_eff_second_stage_vars = len(ep.second_stage_variables)
    num_eff_state_vars = len(ep.state_variables)
    num_dr_vars = len(
        list(generate_all_decision_rule_var_data_objects(working_model))
    )

    # uncertain parameters
    num_uncertain_params = len(working_model.uncertain_params)

    # constraints
    num_cons = len(
        list(working_model.component_data_objects(Constraint, active=True))
    )

    # # equality constraints
    num_eq_cons = len(
        working_model.effective_first_stage_equality_cons
        + working_model.effective_performance_equality_cons
        + working_model.decision_rule_eqns
    )
    num_first_stage_eq_cons = len(working_model.effective_first_stage_equality_cons)
    num_coeff_matching_cons = len(working_model.coefficient_matching_conlist)
    num_other_first_stage_eqns = num_first_stage_eq_cons - num_coeff_matching_cons
    num_performance_eq_cons = len(working_model.effective_performance_equality_cons)
    num_dr_eq_cons = len(working_model.decision_rule_eqns)

    # # inequality constraints
    num_ineq_cons = len(
        working_model.effective_first_stage_inequality_cons
        + working_model.effective_performance_inequality_cons
    )
    num_first_stage_ineq_cons = len(working_model.effective_first_stage_inequality_cons)
    num_performance_ineq_cons = len(working_model.effective_performance_inequality_cons)

    info_log_func = config.progress_logger.info

    IterationLogRecord.log_header_rule(info_log_func)
    info_log_func("Model Statistics:")

    info_log_func(f"  Number of variables : {num_vars}")
    info_log_func(f"    First-stage variables : {num_first_stage_vars}")
    info_log_func(
        f"    Second-stage variables : {num_second_stage_vars} "
        f"({num_eff_second_stage_vars} adj.)"
    )
    info_log_func(
        f"    State variables : {num_state_vars} "
        f"({num_eff_state_vars} adj.)"
    )
    info_log_func(f"    Epigraph variable : {num_epigraph_vars}")
    info_log_func(f"    Decision rule variables : {num_dr_vars}")

    info_log_func(f"  Number of uncertain parameters : {num_uncertain_params}")

    info_log_func(f"  Number of constraints : {num_cons}")
    info_log_func(f"    Equality constraints : {num_eq_cons}")
    info_log_func(f"      Coefficient matching constraints : {num_coeff_matching_cons}")
    info_log_func(f"      Other first-stage equations : {num_other_first_stage_eqns}")
    info_log_func(f"      Performance equations : {num_performance_eq_cons}")
    info_log_func(f"      Decision rule equations : {num_dr_eq_cons}")
    info_log_func(f"    Inequality constraints : {num_ineq_cons}")
    info_log_func(f"      First-stage inequalities : {num_first_stage_ineq_cons}")
    info_log_func(f"      Performance inequalities : {num_performance_ineq_cons}")


def preprocess_model_data(model_data, config, var_partitioning):
    """
    Preprocess model data.
    """
    original_model = model_data.original_model

    # new_preprocess_model_data(model_data, config, var_partitioning)

    # temporary block to track variable partitioning
    # and uncertain parameters after cloning.
    # TODO: model may already have an attribute called `util`;
    #       fix that edge case
    original_model.util = Block(concrete=True)
    original_model.util.first_stage_variables = var_partitioning.first_stage_variables
    original_model.util.second_stage_variables = var_partitioning.second_stage_variables
    original_model.util.state_vars = var_partitioning.state_variables
    original_model.util.uncertain_params = config.uncertain_params

    model_data.util_block = original_model.util

    # keep track of variables after cloning
    cname = unique_component_name(model_data.original_model, 'tmp_var_list')
    src_vars = list(model_data.original_model.component_data_objects(Var))
    setattr(model_data.original_model, cname, src_vars)
    model_data.working_model = model_data.original_model.clone()

    # identify active objective function.
    # (there should only be one at this point)
    # recast to minimization if necessary
    active_objs = list(
        model_data.working_model.component_data_objects(
            Objective, active=True, descend_into=True
        )
    )
    assert len(active_objs) == 1
    active_obj = active_objs[0]
    model_data.active_obj_original_sense = active_obj.sense
    recast_to_min_obj(model_data.working_model, active_obj)

    # === Determine first and second-stage objectives
    identify_objective_functions(model_data.working_model, active_obj)
    active_obj.deactivate()

    # === Put model in standard form
    transform_to_standard_form(model_data.working_model)

    # === Replace variable bounds depending on uncertain params with
    #     explicit inequality constraints
    replace_uncertain_bounds_with_constraints(
        model_data.working_model, model_data.working_model.util.uncertain_params
    )

    # === Add decision rule information
    add_decision_rule_variables(model_data, config)
    add_decision_rule_constraints(model_data, config)

    # === Move bounds on control variables to explicit ineq constraints
    wm_util = model_data.working_model

    # cast bounds on second-stage and state variables to
    # explicit constraints for separation objectives
    for c in model_data.working_model.util.second_stage_variables:
        turn_bounds_to_constraints(c, wm_util, config)
    for c in model_data.working_model.util.state_vars:
        turn_bounds_to_constraints(c, wm_util, config)

    # === Make control_variable_bounds array
    wm_util.ssv_bounds = []
    for c in model_data.working_model.component_data_objects(
        Constraint, descend_into=True
    ):
        if "bound_con" in c.name:
            wm_util.ssv_bounds.append(c)


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

    # initialize first and second-stage summand expressions
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
    current_nl_writer_tol = pyomo_nl_writer.TOL
    pyomo_nl_writer.TOL = 1e-4

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
        pyomo_nl_writer.TOL = current_nl_writer_tol

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
