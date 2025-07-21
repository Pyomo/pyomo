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

'''
Utility functions for the PyROS solver
'''

from collections import namedtuple
from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum, auto
import functools
import itertools as it
import logging
import math
import os
import timeit

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.dependencies import scipy as sp
from pyomo.common.flags import NOTSET
from pyomo.common.errors import ApplicationError, InvalidValueError
from pyomo.common.log import Preformatted
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import HierarchicalTimer, TicTocTimer
from pyomo.core.base import (
    Any,
    Block,
    Component,
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    maximize,
    minimize,
    Param,
    ParamData,
    Reals,
    Var,
    VarData,
    value,
)
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.core.expr.visitor import (
    identify_variables,
    identify_mutable_parameters,
    replace_expressions,
)
from pyomo.core.util import prod
from pyomo.opt import SolverFactory
import pyomo.repn.ampl as pyomo_ampl_repn
from pyomo.repn.parameterized import ParameterizedQuadraticRepnVisitor
import pyomo.repn.plugins.nl_writer as pyomo_nl_writer
from pyomo.repn.util import OrderedVarRecorder
from pyomo.util.vars_from_expressions import get_vars_from_components


# Tolerances used in the code
PARAM_IS_CERTAIN_REL_TOL = 1e-4
PARAM_IS_CERTAIN_ABS_TOL = 0
COEFF_MATCH_REL_TOL = 1e-6
COEFF_MATCH_ABS_TOL = 0
ABS_CON_CHECK_FEAS_TOL = 1e-5
PRETRIANGULAR_VAR_COEFF_TOL = 1e-6
POINT_IN_UNCERTAINTY_SET_TOL = 1e-8
DR_POLISHING_PARAM_PRODUCT_ZERO_TOL = 1e-10

TIC_TOC_SOLVE_TIME_ATTR = "pyros_tic_toc_time"
DEFAULT_LOGGER_NAME = "pyomo.contrib.pyros"
DEFAULT_SEPARATION_PRIORITY = 0
BYPASSING_SEPARATION_PRIORITY = None


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
    """Starts timer at entry, stores elapsed time at exit.

    Parameters
    ----------
    timing_data_obj : TimingData
        Timing data object.

    code_block_name : str
        Name of code block being timed.

    is_main_timer : bool
        If ``is_main_timer=True``, the start time is stored in the
        timing_data_obj, allowing calculation of total elapsed time 'on
        the fly' (e.g. to enforce a time limit) using
        ``get_main_elapsed_time(timing_data_obj)``.

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


def standardize_component_data(
    obj,
    valid_ctype,
    valid_cdatatype,
    ctype_validator=None,
    cdatatype_validator=None,
    allow_repeats=False,
    from_iterable=None,
):
    """
    Cast an object to a list of Pyomo ComponentData objects.

    Parameters
    ----------
    obj : Component, ComponentData, or iterable
        Object from which component data objects
        are cast.
    valid_ctype : type or tuple of type
        Valid Component type(s).
    valid_cdatatype : type or tuple of type
        Valid ComponentData type(s).
    ctype_validator : None or callable, optional
        Validator for component objects derived from `obj`.
    cdatatype_validator : None or callable, optional
        Validator for component data objects derived from `obj`.
    allow_repeats : bool, optional
        True to allow for nonunique component data objects
        derived from `obj`, False otherwise.
    from_iterable : str, optional
        Description of the object to include in error messages.
        Meant to be used if the object is an iterable from which
        to derive component data objects.

    Returns
    -------
    list of ComponentData
        The ComponentData objects derived from `obj`.
        Note: If `obj` is a valid ComponentData type,
        then ``[obj]`` is returned.

    Raises
    ------
    TypeError
        If `obj` is not an iterable and not an instance of
        `valid_ctype` or `valid_cdatatype`.
    ValueError
        If ``allow_repeats=False`` and there are duplicates
        among the component data objects derived from `obj`.
    """
    if isinstance(obj, valid_ctype):
        if ctype_validator is not None:
            ctype_validator(obj)
        ans = list(obj.values())
        if cdatatype_validator is not None:
            for entry in ans:
                cdatatype_validator(entry)
        return ans
    elif isinstance(obj, valid_cdatatype):
        if cdatatype_validator is not None:
            cdatatype_validator(obj)
        return [obj]
    elif isinstance(obj, Component):
        # deal with this case separately from general
        # iterables to prevent iteration over an invalid
        # component type
        raise TypeError(
            f"Input object {obj!r} "
            "is not of valid component type "
            f"{valid_ctype.__name__} or component data type "
            f"(got type {type(obj).__name__})."
        )
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        ans = []
        for item in obj:
            ans.extend(
                standardize_component_data(
                    item,
                    valid_ctype=valid_ctype,
                    valid_cdatatype=valid_cdatatype,
                    ctype_validator=ctype_validator,
                    cdatatype_validator=cdatatype_validator,
                    allow_repeats=allow_repeats,
                    from_iterable=obj,
                )
            )
    else:
        from_iterable_qual = (
            f" (entry of iterable {from_iterable})" if from_iterable is not None else ""
        )
        raise TypeError(
            f"Input object {obj!r}{from_iterable_qual} "
            "is not of valid component type "
            f"{valid_ctype.__name__} or component data type "
            f"{valid_cdatatype.__name__} (got type {type(obj).__name__})."
        )

    # check for duplicates if desired
    if not allow_repeats and len(ans) != len(ComponentSet(ans)):
        comp_name_list = [comp.name for comp in ans]
        raise ValueError(
            f"Standardized component list {comp_name_list} "
            f"derived from input {obj} "
            "contains duplicate entries."
        )

    return ans


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
    Check that the partitioning of the in-scope variables of the
    model is valid.

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

    # uncertain parameters can be VarData objects;
    # ensure they are not considered decision variables here
    active_model_vars = ComponentSet(
        get_vars_from_components(
            block=model,
            active=True,
            include_fixed=True,
            descend_into=True,
            ctype=(Objective, Constraint),
        )
    ) - ComponentSet(config.uncertain_params)
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

    first_stage_vars = ComponentSet(config.first_stage_variables) & active_model_vars
    second_stage_vars = ComponentSet(config.second_stage_variables) & active_model_vars
    state_vars = active_model_vars - (first_stage_vars | second_stage_vars)

    return VariablePartitioning(
        list(first_stage_vars), list(second_stage_vars), list(state_vars)
    )


def _get_uncertain_param_val(var_or_param_data):
    """
    Get value of VarData/ParamData object
    that is considered an uncertain parameter.

    For any unfixed VarData object, we assume that
    the `lower` and `upper` attributes are identical,
    so the value of `lower` is returned in lieu of
    the level value.

    Parameters
    ----------
    var_or_param_data : VarData or ParamData
        Object to be evaluated.

    Returns
    -------
    object
        Value of the VarData/ParamData object.
        The value is typically of a numeric type.
    """
    if isinstance(var_or_param_data, ParamData):
        expr_to_evaluate = var_or_param_data
    elif isinstance(var_or_param_data, VarData):
        if var_or_param_data.fixed:
            expr_to_evaluate = var_or_param_data
        else:
            expr_to_evaluate = var_or_param_data.lower
    else:
        raise ValueError(
            f"Uncertain parameter object {var_or_param_data!r}"
            f"is of type {type(var_or_param_data).__name__!r}, "
            "but should be of type "
            f"{ParamData.__name__} or {VarData.__name__}."
        )

    return value(expr_to_evaluate, exception=True)


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

        - there are entries of `config.uncertain_params`
          that are also in `config.first_stage_variables` or
          `config.second_stage_variables`
        - dimension of uncertainty set does not equal number of
          uncertain parameters
        - uncertainty set `validate()` method fails.
        - nominal parameter realization is not in the uncertainty set.
    """
    check_components_descended_from_model(
        model=model,
        components=config.uncertain_params,
        components_name="uncertain parameters",
        config=config,
    )

    first_stg_vars = config.first_stage_variables
    second_stg_vars = config.second_stage_variables
    for stg_str, vars in zip(["first", "second"], [first_stg_vars, second_stg_vars]):
        overlapping_uncertain_params = ComponentSet(vars) & ComponentSet(
            config.uncertain_params
        )
        if overlapping_uncertain_params:
            overlapping_var_list = "\n ".join(
                f"{var.name!r}" for var in overlapping_uncertain_params
            )
            config.progress_logger.error(
                f"The following Vars were found in both `{stg_str}_stage_variables`"
                f"and `uncertain_params`:\n {overlapping_var_list}"
                "\nEnsure no Vars are included in both arguments."
            )
            raise ValueError(
                f"Arguments `{stg_str}_stage_variables` and `uncertain_params` "
                "contain at least one common Var object."
            )

    if len(config.uncertain_params) != config.uncertainty_set.dim:
        raise ValueError(
            "Length of argument `uncertain_params` does not match dimension "
            "of argument `uncertainty_set` "
            f"({len(config.uncertain_params)} != {config.uncertainty_set.dim})."
        )

    # fill-in nominal point as necessary, if not provided.
    # otherwise, check length matches uncertainty dimension
    if not config.nominal_uncertain_param_vals:
        config.nominal_uncertain_param_vals = [
            # NOTE: this allows uncertain parameters that are of type
            #       VarData and implicitly fixed by identical bounds
            #       that are mutable expressions in ParamData-type
            #       uncertain parameters;
            #       the bounds expressions are evaluated to
            #       to get the nominal realization
            _get_uncertain_param_val(param)
            for param in config.uncertain_params
        ]
    elif len(config.nominal_uncertain_param_vals) != len(config.uncertain_params):
        raise ValueError(
            "Lengths of arguments `uncertain_params` and "
            "`nominal_uncertain_param_vals` "
            "do not match "
            f"({len(config.uncertain_params)} != "
            f"{len(config.nominal_uncertain_param_vals)})."
        )

    # validate uncertainty set
    config.uncertainty_set.validate(config=config)

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


class ModelData:
    """
    Container for modeling objects from which the PyROS
    subproblems are constructed.

    Parameters
    ----------
    original_model : ConcreteModel
        Original user-provided model.
    timing : TimingData
        Main timing data object.

    Attributes
    ----------
    original_model : ConcreteModel
        Original user-provided model.
    timing : TimingData
        Main PyROS solver timing data object.
    working_model : ConcreteModel
        Preprocessed clone of `original_model` from which
        the PyROS cutting set subproblems are to be
        constructed.
    separation_priority_order : dict
        Mapping from constraint names to separation priority
        values.
    separation_priority_suffix_finder : SuffixFinder
        Object for resolving active Suffix components
        added to the model by the user as a means of
        prioritizing separation problems mapped to
        second-stage inequality constraints.
    """

    def __init__(self, original_model, config, timing):
        self.original_model = original_model
        self.timing = timing
        self.config = config
        self.separation_priority_order = dict()
        # working model will be addressed by preprocessing
        self.working_model = None
        self.separation_priority_suffix_finder = SuffixFinder(
            name="pyros_separation_priority", default=NOTSET
        )

    def preprocess(self, user_var_partitioning):
        """
        Preprocess model data.

        See :meth:`~preprocess_model_data`.

        Returns
        -------
        bool
            True if robust infeasibility detected, False otherwise.
        """
        return preprocess_model_data(self, user_var_partitioning)

    def get_user_separation_priority(self, component_data, component_data_name):
        """
        Infer user specification for the separation priority/priorities
        of the second-stage inequality constraint/constraints derived
        from a given component data attribute of the working model.

        Parameters
        ----------
        component_data : ComponentData
            Component data from which the inequality constraints
            are meant to be derived.
        component_data_name : str
            Name of the component data object as it is expected
            to appear in ``self.config.separation_priority_order``.

        Returns
        -------
        numeric type or None
            Priority of the derived constraint(s).

        Notes
        -----
        The separation priorities for the constraints derived from
        ``component_data`` are inferred from either the
        active ``Suffix`` components of ``self.working_model``
        with name 'pyros_separation_priority'
        or from ``self.config.separation_priority``.
        Priorities specified through the active ``Suffix`` components
        take precedence over priorities specified through
        ``self.config.separation_priority_order``.
        Moreover, priorities are inferred from ``Suffix`` components
        using the Pyomo ``SuffixFinder``.
        """
        priority = self.separation_priority_suffix_finder.find(component_data)
        if priority is NOTSET:
            priority = self.config.separation_priority_order.get(
                component_data_name, DEFAULT_SEPARATION_PRIORITY
            )
        return priority


def setup_quadratic_expression_visitor(
    wrt, subexpression_cache=None, var_map=None, var_order=None, sorter=None
):
    """Setup a parameterized quadratic expression walker."""
    visitor = ParameterizedQuadraticRepnVisitor(
        subexpression_cache={} if subexpression_cache is None else subexpression_cache,
        var_recorder=OrderedVarRecorder(
            var_map={} if var_map is None else var_map,
            var_order={} if var_order is None else var_order,
            sorter=sorter,
        ),
        wrt=wrt,
    )
    visitor.expand_nonlinear_products = True
    return visitor


class BoundType:
    """
    Indicator for whether a bound on a variable/constraint
    is a lower bound, "equality" bound, or upper bound.
    """

    LOWER = "lower"
    EQ = "eq"
    UPPER = "upper"


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
    domain_bound, declared_bound, uncertain_params, bound_type
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
    bound_type : {BoundType.LOWER, BoundType.UPPER}
        Indication of whether the domain bound and declared bound
        specify lower or upper bounds for the variable value.

    Returns
    -------
    certain_bound : numeric type, NumericExpression, or None
        Bound that independent of the uncertain parameters.
    uncertain_bound : numeric expression or None
        Bound that is dependent on the uncertain parameters.
    """
    if bound_type not in {BoundType.LOWER, BoundType.UPPER}:
        raise ValueError(
            f"Argument {bound_type=!r} should be either "
            f"'{BoundType.LOWER}' or '{BoundType.UPPER}'."
        )

    if declared_bound is not None:
        uncertain_params_in_declared_bound = ComponentSet(
            uncertain_params
        ) & ComponentSet(identify_mutable_parameters(declared_bound))
    else:
        uncertain_params_in_declared_bound = False

    if not uncertain_params_in_declared_bound:
        uncertain_bound = None

        if declared_bound is None:
            certain_bound = domain_bound
        elif domain_bound is None:
            certain_bound = declared_bound
        else:
            if bound_type == BoundType.LOWER:
                certain_bound = (
                    declared_bound
                    if value(declared_bound) >= domain_bound
                    else domain_bound
                )
            else:
                certain_bound = (
                    declared_bound
                    if value(declared_bound) <= domain_bound
                    else domain_bound
                )
    else:
        uncertain_bound = declared_bound
        certain_bound = domain_bound

    return certain_bound, uncertain_bound


BoundTriple = namedtuple(
    "BoundTriple", (BoundType.LOWER, BoundType.EQ, BoundType.UPPER)
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
        bound_type=BoundType.LOWER,
    )
    certain_ub, uncertain_ub = determine_certain_and_uncertain_bound(
        domain_bound=domain_ub,
        declared_bound=declared_ub,
        uncertain_params=uncertain_params,
        bound_type=BoundType.UPPER,
    )

    certain_bounds = rearrange_bound_pair_to_triple(
        lower_bound=certain_lb, upper_bound=certain_ub
    )
    uncertain_bounds = rearrange_bound_pair_to_triple(
        lower_bound=uncertain_lb, upper_bound=uncertain_ub
    )

    return certain_bounds, uncertain_bounds


def get_effective_var_partitioning(model_data):
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

    Returns
    -------
    effective_partitioning : VariablePartitioning
        Effective variable partitioning.
    """
    config = model_data.config
    working_model = model_data.working_model
    user_var_partitioning = model_data.working_model.user_var_partitioning

    # truly nonadjustable variables
    nonadjustable_var_set = ComponentSet()

    effective_uncertain_params_set = ComponentSet(
        working_model.effective_uncertain_params
    )
    if not effective_uncertain_params_set:
        config.progress_logger.info(
            "Model has no effective uncertain parameters. "
            "All variables are considered effectively first-stage."
        )
        return VariablePartitioning(
            first_stage_variables=(
                user_var_partitioning.first_stage_variables
                + user_var_partitioning.second_stage_variables
                + user_var_partitioning.state_variables
            ),
            second_stage_variables=[],
            state_variables=[],
        )

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
                wvar, working_model.effective_uncertain_params
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
                config.progress_logger.debug(" the variable is fixed explicitly")

            if certain_var_bounds.eq is not None:
                config.progress_logger.debug(" the variable is fixed by domain/bounds")

    # determine constraints that are potentially applicable for
    # pretriangularization
    certain_eq_cons = ComponentSet()
    for wcon in working_model.component_data_objects(Constraint, active=True):
        if not wcon.equality:
            continue
        uncertain_params_in_expr = (
            ComponentSet(identify_mutable_parameters(wcon.expr))
            & effective_uncertain_params_set
        )
        if uncertain_params_in_expr:
            continue
        certain_eq_cons.add(wcon)

    pretriangular_con_var_map = ComponentMap()
    for num_passes in it.count(1):
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
                visitor = setup_quadratic_expression_visitor(wrt=[])
                ccon_expr_repn = visitor.walk_expression(expr=ccon.body - ccon.upper)
                adj_var_appears_linearly = adj_var_in_con not in ComponentSet(
                    identify_variables(ccon_expr_repn.nonlinear)
                ) and id(adj_var_in_con) in ComponentSet(ccon_expr_repn.linear)
                if adj_var_appears_linearly:
                    adj_var_linear_coeff = ccon_expr_repn.linear[id(adj_var_in_con)]
                    if abs(adj_var_linear_coeff) > PRETRIANGULAR_VAR_COEFF_TOL:
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
        effective_first_stage_vars + effective_second_stage_vars + effective_state_vars
    )

    config.progress_logger.debug("Effective partitioning statistics:")
    config.progress_logger.debug(f"  Variables: {num_vars}")
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


def add_effective_var_partitioning(model_data):
    """
    Obtain a repartitioning of the in-scope variables of the
    working model according to known adjustability to the
    uncertain parameters, and add this repartitioning to the
    working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    """
    effective_partitioning = get_effective_var_partitioning(model_data)
    model_data.working_model.effective_var_partitioning = VariablePartitioning(
        **effective_partitioning._asdict()
    )


def create_bound_constraint_expr(expr, bound, bound_type, standardize=True):
    """
    Create a relational expression establishing a bound
    for a numeric expression of interest.

    If desired, the expression is such that `bound` appears on the
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
    bound_type : BoundType
        Indicator for whether `expr` is to be lower bounded,
        equality bounded, or upper bounded, by `bound`.
    standardize : bool, optional
        True to ensure `expr` appears on the left-hand side of the
        relational operator, False otherwise.

    Returns
    -------
    RelationalExpression
        Establishes a bound on `expr`.
    """
    if bound_type == BoundType.LOWER:
        return -expr <= -bound if standardize else bound <= expr
    elif bound_type == BoundType.EQ:
        return expr == bound
    elif bound_type == BoundType.UPPER:
        return expr <= bound
    else:
        raise ValueError(f"Bound type {bound_type!r} not supported.")


def remove_var_declared_bound(var, bound_type):
    """
    Remove the specified declared bound(s) of a variable data object.

    Parameters
    ----------
    var : VarData
        Variable data object of interest.
    bound_type : BoundType
        Indicator for the declared bound(s) to remove.
        Note: if BoundType.EQ is specified, then both the
        lower and upper bounds are removed.
    """
    if bound_type == BoundType.LOWER:
        var.setlb(None)
    elif bound_type == BoundType.EQ:
        var.setlb(None)
        var.setub(None)
    elif bound_type == BoundType.UPPER:
        var.setub(None)
    else:
        raise ValueError(
            f"Bound type {bound_type!r} not supported. "
            f"Bound type must be '{BoundType.LOWER}', "
            f"'{BoundType.EQ}, or '{BoundType.UPPER}'."
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
    in this method are considered second-stage constraints.

    Uncertain bounds that have been assigned a separation priority
    of None are not reformulated, and are subsequently enforced
    subject to only the nominal uncertain parameter realization.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    """
    working_model = model_data.working_model
    nonadjustable_vars = working_model.effective_var_partitioning.first_stage_variables
    uncertain_params_set = ComponentSet(working_model.effective_uncertain_params)
    for var in nonadjustable_vars:
        _, declared_bounds = get_var_bound_pairs(var)
        declared_bound_triple = rearrange_bound_pair_to_triple(*declared_bounds)
        var_name = var.getname(
            relative_to=working_model.user_model, fully_qualified=True
        )
        var_bound_sep_priority = model_data.get_user_separation_priority(
            component_data=var, component_data_name=var_name
        )
        for btype, bound in declared_bound_triple._asdict().items():
            is_bound_uncertain = bound is not None and (
                ComponentSet(identify_mutable_parameters(bound)) & uncertain_params_set
            )
            is_bound_second_stage = (
                is_bound_uncertain
                and var_bound_sep_priority is not BYPASSING_SEPARATION_PRIORITY
            )
            if is_bound_second_stage:
                new_con_expr = create_bound_constraint_expr(var, bound, btype)
                new_con_name = f"var_{var_name}_uncertain_{btype}_bound_con"
                remove_var_declared_bound(var, btype)
                if btype == BoundType.EQ:
                    working_model.second_stage.equality_cons[new_con_name] = (
                        new_con_expr
                    )
                else:
                    working_model.second_stage.inequality_cons[new_con_name] = (
                        new_con_expr
                    )
                model_data.separation_priority_order[new_con_name] = (
                    var_bound_sep_priority
                )


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
    they are taken to be (effective) second-stage constraints.

    Bounds that have been assigned a separation priority
    of None are reformulated to first-stage constraints,
    and are subsequently enforced subject to only the
    nominal uncertain parameter realization.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    """
    working_model = model_data.working_model

    adjustable_vars = (
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    for var in adjustable_vars:
        cert_bound_triple, uncert_bound_triple = get_var_certain_uncertain_bounds(
            var, working_model.effective_uncertain_params
        )
        var_name = var.getname(
            relative_to=working_model.user_model, fully_qualified=True
        )
        cert_uncert_bound_zip = (
            ("certain", cert_bound_triple),
            ("uncertain", uncert_bound_triple),
        )
        var_bound_sep_priority = model_data.get_user_separation_priority(
            component_data=var, component_data_name=var_name
        )
        is_bound_second_stage = (
            var_bound_sep_priority is not BYPASSING_SEPARATION_PRIORITY
        )
        if is_bound_second_stage:
            ineq_con_component = working_model.second_stage.inequality_cons
            eq_con_component = working_model.second_stage.equality_cons
        else:
            ineq_con_component = working_model.first_stage.inequality_cons
            eq_con_component = working_model.first_stage.dr_independent_equality_cons
        for certainty_desc, bound_triple in cert_uncert_bound_zip:
            for btype, bound in bound_triple._asdict().items():
                if bound is not None:
                    new_con_name = f"var_{var_name}_{certainty_desc}_{btype}_bound_con"
                    new_con_expr = create_bound_constraint_expr(var, bound, btype)
                    if btype == BoundType.EQ:
                        eq_con_component[new_con_name] = new_con_expr
                    else:
                        ineq_con_component[new_con_name] = new_con_expr

                    if is_bound_second_stage:
                        model_data.separation_priority_order[new_con_name] = (
                            var_bound_sep_priority
                        )

        remove_all_var_bounds(var)


def _replace_vars_in_component_exprs(block, substitution_map, ctype):
    """
    Substitute other objects for Vars in the expression attributes
    of the component objects of a given type in a given block.

    For efficiency purposes, only components whose expressions
    contain the Vars to remove via the substitution are acted upon.

    Named expressions in the components acted upon are descended
    into, but not removed.

    Parameters
    ----------
    block : BlockData
        Block on which to perform the replacement.
    substitution_map : ComponentMap
        First entry of each tuple is a Var to remove,
        second entry is an object to introduce in its place.
    ctype : type or tuple of type
        Type(s) of the components whose expressions are to be
        modified.
    """
    vars_to_be_replaced = ComponentSet([var for var, _ in substitution_map.items()])
    substitution_map = {id(var): dest for var, dest in substitution_map.items()}
    for cdata in block.component_data_objects(ctype, active=None, descend_into=True):
        # efficiency: act only on components containing
        #             the Vars to be substituted
        if ComponentSet(identify_variables(cdata.expr)) & vars_to_be_replaced:
            cdata.set_value(
                replace_expressions(
                    expr=cdata.expr,
                    substitution_map=substitution_map,
                    descend_into_named_expressions=True,
                    remove_named_expressions=False,
                )
            )


def replace_vars_with_params(block, var_to_param_map):
    """
    Substitute ParamData objects for VarData objects
    in the Expression, Constraint, and Objective components
    declared on a block and all its sub-blocks.

    Note that when performing the substitutions in the
    Constraint and Objective components,
    named Expressions are descended into, but not replaced.

    Parameters
    ----------
    block : BlockData
        Block on which to perform the substitution.
    var_to_param_map : ComponentMap
        Mapping from VarData objects to be replaced
        to the ParamData objects to be introduced.
    """
    _replace_vars_in_component_exprs(
        block=block,
        substitution_map=var_to_param_map,
        ctype=(Expression, Constraint, Objective),
    )


def setup_working_model(model_data, user_var_partitioning):
    """
    Set up (construct) the working model based on user inputs,
    and add it to the model data object.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    user_var_partitioning : VariablePartitioning
        User-based partitioning of the in-scope
        variables of the input model.
    """
    config = model_data.config
    original_model = model_data.original_model

    # add temporary block to help keep track of variables
    # and uncertain parameters after cloning
    temp_util_block_attr_name = unique_component_name(original_model, "util")
    original_model.add_component(temp_util_block_attr_name, Block())
    orig_temp_util_block = getattr(original_model, temp_util_block_attr_name)
    orig_temp_util_block.orig_uncertain_params = config.uncertain_params
    orig_temp_util_block.user_var_partitioning = VariablePartitioning(
        **user_var_partitioning._asdict()
    )

    # now set up working model
    model_data.working_model = working_model = ConcreteModel()

    # stagewise blocks for containing stagewise constraints
    working_model.first_stage = Block()
    working_model.first_stage.dr_independent_equality_cons = Constraint(Any)
    working_model.first_stage.dr_dependent_equality_cons = Constraint(Any)
    working_model.first_stage.inequality_cons = Constraint(Any)
    working_model.second_stage = Block()
    working_model.second_stage.equality_cons = Constraint(Any)
    working_model.second_stage.inequality_cons = Constraint(Any)

    # original user model will be a sub-block of working model,
    # in order to avoid attribute name clashes later
    working_model.user_model = original_model.clone()

    # facilitate later retrieval of the user var partitioning
    working_temp_util_block = getattr(
        working_model.user_model, temp_util_block_attr_name
    )
    model_data.working_model.orig_uncertain_params = (
        working_temp_util_block.orig_uncertain_params.copy()
    )
    working_model.user_var_partitioning = VariablePartitioning(
        **working_temp_util_block.user_var_partitioning._asdict()
    )

    # we are done with the util blocks
    delattr(original_model, temp_util_block_attr_name)
    delattr(working_model.user_model, temp_util_block_attr_name)

    uncertain_param_var_idxs = []
    for idx, obj in enumerate(working_model.orig_uncertain_params):
        if isinstance(obj, VarData):
            obj.fix()
            uncertain_param_var_idxs.append(idx)
    temp_params = working_model.temp_uncertain_params = Param(
        uncertain_param_var_idxs,
        within=Reals,
        initialize={
            idx: config.nominal_uncertain_param_vals[idx]
            for idx in uncertain_param_var_idxs
        },
        mutable=True,
    )
    working_model.uncertain_params = [
        temp_params[idx] if idx in uncertain_param_var_idxs else orig_param
        for idx, orig_param in enumerate(working_model.orig_uncertain_params)
    ]

    # don't want to pass over the model components unless
    # at least one Var is to be replaced
    if uncertain_param_var_idxs:
        uncertain_var_to_param_map = ComponentMap(
            (working_model.orig_uncertain_params[idx], temp_param)
            for idx, temp_param in temp_params.items()
        )
        replace_vars_with_params(
            working_model, var_to_param_map=uncertain_var_to_param_map
        )
        for var, param in uncertain_var_to_param_map.items():
            config.progress_logger.debug(
                "Uncertain parameter with name "
                f"{var.name!r} (relative to the working model clone) "
                f"is of type {VarData.__name__}. "
                f"A newly declared {ParamData.__name__} object "
                f"with name {param.name!r} "
                f"has been substituted for the {VarData.__name__} object "
                "in all named expressions, constraints, and objectives "
                "of the working model clone. "
            )

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


def standardize_inequality_constraints(model_data):
    """
    Standardize the inequality constraints of the working model,
    and classify them as first-stage inequalities or second-stage
    inequalities.

    Parameters
    ----------
    model_data : model data object
        Main model data object, containing the working model.
    """
    working_model = model_data.working_model
    uncertain_params_set = ComponentSet(working_model.effective_uncertain_params)
    adjustable_vars_set = ComponentSet(
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    for con in working_model.original_active_inequality_cons:
        uncertain_params_in_con_expr = (
            ComponentSet(identify_mutable_parameters(con.expr)) & uncertain_params_set
        )
        adjustable_vars_in_con_body = (
            ComponentSet(identify_variables(con.body)) & adjustable_vars_set
        )
        con_rel_name = con.getname(
            relative_to=working_model.user_model, fully_qualified=True
        )
        con_sep_priority = model_data.get_user_separation_priority(
            component_data=con, component_data_name=con_rel_name
        )
        is_con_potentially_second_stage = (
            uncertain_params_in_con_expr | adjustable_vars_in_con_body
        ) and con_sep_priority is not BYPASSING_SEPARATION_PRIORITY
        if is_con_potentially_second_stage:
            con_bounds_triple = rearrange_bound_pair_to_triple(
                lower_bound=con.lower, upper_bound=con.upper
            )
            finite_bounds = {
                btype: bd
                for btype, bd in con_bounds_triple._asdict().items()
                if bd is not None
            }
            for btype, bound in finite_bounds.items():
                if btype == BoundType.EQ:
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

                std_con_expr = create_bound_constraint_expr(
                    expr=con.body, bound=bound, bound_type=btype, standardize=True
                )
                new_con_name = f"ineq_con_{con_rel_name}_{btype}_bound_con"

                uncertain_params_in_std_expr = uncertain_params_set & ComponentSet(
                    identify_mutable_parameters(std_con_expr)
                )
                if adjustable_vars_in_con_body | uncertain_params_in_std_expr:
                    working_model.second_stage.inequality_cons[new_con_name] = (
                        std_con_expr
                    )
                    # account for user-specified priority specifications
                    model_data.separation_priority_order[new_con_name] = (
                        con_sep_priority
                    )
                else:
                    # we do not want to modify the arrangement of
                    # lower bound for first-stage inequalities, so
                    # pass `standardize=False`
                    working_model.first_stage.inequality_cons[new_con_name] = (
                        create_bound_constraint_expr(
                            expr=con.body,
                            bound=bound,
                            bound_type=btype,
                            standardize=False,
                        )
                    )

            # constraint has now been moved over to stagewise blocks
            con.deactivate()
        else:
            # constraint depends on the nonadjustable variables only
            working_model.first_stage.inequality_cons[f"ineq_con_{con_rel_name}"] = (
                con.expr
            )
            con.deactivate()


def standardize_equality_constraints(model_data):
    """
    Classify the original active equality constraints of the
    working model as first-stage or second-stage constraints.

    Parameters
    ----------
    model_data : model data object
        Main model data object, containing the working model.
    """
    working_model = model_data.working_model
    uncertain_params_set = ComponentSet(working_model.effective_uncertain_params)
    adjustable_vars_set = ComponentSet(
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    for con in working_model.original_active_equality_cons:
        uncertain_params_in_con_expr = (
            ComponentSet(identify_mutable_parameters(con.expr)) & uncertain_params_set
        )
        adjustable_vars_in_con_body = (
            ComponentSet(identify_variables(con.body)) & adjustable_vars_set
        )

        # note: none of the equality constraint expressions are modified
        con_rel_name = con.getname(
            relative_to=working_model.user_model, fully_qualified=True
        )
        con_sep_priority = model_data.get_user_separation_priority(
            component_data=con, component_data_name=con_rel_name
        )
        new_con_name = f"eq_con_{con_rel_name}"
        is_con_second_stage = (
            uncertain_params_in_con_expr | adjustable_vars_in_con_body
        ) and con_sep_priority is not BYPASSING_SEPARATION_PRIORITY
        if is_con_second_stage:
            working_model.second_stage.equality_cons[new_con_name] = con.expr
            model_data.separation_priority_order[new_con_name] = con_sep_priority
        else:
            working_model.first_stage.dr_independent_equality_cons[new_con_name] = (
                con.expr
            )

        # definitely don't want active duplicate
        con.deactivate()


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
    uncertain_param_set = ComponentSet(working_model.effective_uncertain_params)

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


def standardize_active_objective(model_data):
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
    variables and the effective uncertain parameters.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    """
    config = model_data.config
    working_model = model_data.working_model

    active_obj = next(
        working_model.component_data_objects(Objective, active=True, descend_into=True)
    )
    model_data.active_obj_original_sense = active_obj.sense

    # per-stage summands will be useful for reporting later
    declare_objective_expressions(working_model=working_model, objective=active_obj)

    # useful for later
    working_model.first_stage.epigraph_var = Var(
        initialize=value(active_obj, exception=False)
    )

    # we add the epigraph objective later, as needed,
    # on a per subproblem basis;
    # doing so is more efficient than adding the objective now
    active_obj.deactivate()

    # add the epigraph constraint
    adjustable_vars = (
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )
    uncertain_params_in_obj = ComponentSet(
        identify_mutable_parameters(active_obj.expr)
    ) & ComponentSet(working_model.effective_uncertain_params)
    adjustable_vars_in_obj = (
        ComponentSet(identify_variables(active_obj.expr)) & adjustable_vars
    )
    if uncertain_params_in_obj | adjustable_vars_in_obj:
        if config.objective_focus == ObjectiveType.worst_case:
            working_model.second_stage.inequality_cons["epigraph_con"] = (
                working_model.full_objective.expr
                - working_model.first_stage.epigraph_var
                <= 0
            )
            model_data.separation_priority_order["epigraph_con"] = (
                DEFAULT_SEPARATION_PRIORITY
            )
        elif config.objective_focus == ObjectiveType.nominal:
            working_model.first_stage.inequality_cons["epigraph_con"] = (
                working_model.full_objective.expr
                - working_model.first_stage.epigraph_var
                <= 0
            )
        else:
            raise ValueError(
                "Classification of the epigraph constraint with uncertain "
                "and/or adjustable components not implemented "
                f"for objective focus {config.objective_focus!r}."
            )
    else:
        working_model.first_stage.inequality_cons["epigraph_con"] = (
            working_model.full_objective.expr - working_model.first_stage.epigraph_var
            <= 0
        )


def get_all_nonadjustable_variables(working_model):
    """
    Get all nonadjustable variables of the working model.

    The nonadjustable variables comprise the:

    - epigraph variable
    - decision rule variables
    - effective first-stage variables
    """
    epigraph_var = working_model.first_stage.epigraph_var
    decision_rule_vars = list(
        generate_all_decision_rule_var_data_objects(working_model)
    )
    effective_first_stage_vars = (
        working_model.effective_var_partitioning.first_stage_variables
    )

    return [epigraph_var] + decision_rule_vars + effective_first_stage_vars


def get_all_first_stage_eq_cons(working_model):
    return list(working_model.first_stage.dr_dependent_equality_cons.values()) + list(
        working_model.first_stage.dr_independent_equality_cons.values()
    )


def get_all_adjustable_variables(working_model):
    """
    Get all variables considered adjustable.
    """
    return (
        working_model.effective_var_partitioning.second_stage_variables
        + working_model.effective_var_partitioning.state_variables
    )


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
    for indexed_var in working_blk.first_stage.decision_rule_vars:
        yield from indexed_var.values()


def generate_all_decision_rule_eqns(working_blk):
    """
    Generate sequence of all decision rule equations.
    """
    yield from working_blk.second_stage.decision_rule_eqns.values()


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


def get_dr_var_to_monomial_map(working_blk):
    """
    Get mapping from all decision rule variables in the working
    block to their corresponding DR equation monomials.

    Parameters
    ----------
    working_blk : BlockData
        Working model Block, containing the decision rule
        components.

    Returns
    -------
    ComponentMap
        The desired mapping.
    """
    dr_var_to_monomial_map = ComponentMap()
    for ss_var in working_blk.effective_var_partitioning.second_stage_variables:
        dr_expr = get_dr_expression(working_blk, ss_var)
        for dr_monomial in dr_expr.args:
            if dr_monomial.is_expression_type():
                # degree > 1 monomial expression of form
                # (product of uncertain params) * dr variable
                dr_var_in_term = dr_monomial.args[-1]
            else:
                # the static term (intercept)
                dr_var_in_term = dr_monomial

            dr_var_to_monomial_map[dr_var_in_term] = dr_monomial

    return dr_var_to_monomial_map


def check_time_limit_reached(timing_data, config):
    """
    Return true if the PyROS solver time limit is reached,
    False otherwise.

    Returns
    -------
    bool
        True if time limit reached, False otherwise.
    """
    return (
        config.time_limit is not None
        and timing_data.get_main_elapsed_time() >= config.time_limit
    )


def _reformulate_eq_con_scenario_uncertainty(
    model_data,
    discrete_set,
    ss_eq_con,
    ss_eq_con_index,
    ss_var_id_to_dr_expr_map,
    all_dr_vars_set,
):
    """
    Reformulate a second-stage equality constraint that
    is independent of the state variables and subject
    to scenario-based uncertainty.

    This reformulation merely involves adding to the set
    of first-stage equalities the original constraint
    subject to each (hard-coded) scenario in the uncertainty set.

    The original constraint is removed from the model.

    Parameters
    ----------
    model_data : ModelData
        Model data object, with mostly preprocessed working model.
    discrete_set : UncertaintySet
        Uncertainty set with scenario-based geometry.
    ss_eq_con : ConstraintData
        Second-stage equality constraint to be reformulated.
        Expected to be a member of
        ``model_data.working_model.second_stage.equality_cons``.
    ss_eq_con_index : hashable
        Index of the equality constraint in
        ``model_data.working_model.second_stage.equality_cons``.
    ss_var_id_to_dr_expr_map : dict
        Mapping from object IDs of second-stage variables to
        corresponding decision rule expressions.
    all_dr_vars_set : ComponentSet
        A set of all the decision rule variables declared on
        ``model_data.working_model``.
    """
    working_model = model_data.working_model
    con_expr_after_dr_substitution = replace_expressions(
        expr=ss_eq_con.expr, substitution_map=ss_var_id_to_dr_expr_map
    )
    vars_in_coeff_expr = ComponentSet(
        identify_variables(con_expr_after_dr_substitution)
    )
    has_con_dr_vars = vars_in_coeff_expr & all_dr_vars_set
    indexed_con_to_update = (
        working_model.first_stage.dr_dependent_equality_cons
        if has_con_dr_vars
        else working_model.first_stage.dr_independent_equality_cons
    )

    scenarios_enum = enumerate(discrete_set.scenarios)
    for sc_idx, scenario in scenarios_enum:
        indexed_con_to_update[f"scenario_{sc_idx}_{ss_eq_con_index}"] = (
            replace_expressions(
                expr=con_expr_after_dr_substitution,
                substitution_map={
                    id(param): scenario_val
                    for param, scenario_val in zip(
                        working_model.uncertain_params, scenario
                    )
                },
            )
        )
    del working_model.second_stage.equality_cons[ss_eq_con_index]


def _reformulate_eq_con_continuous_uncertainty(
    model_data,
    config,
    ss_eq_con,
    ss_eq_con_index,
    ss_var_id_to_dr_expr_map,
    uncertain_param_id_to_temp_var_map,
    originally_unfixed_vars,
    all_dr_vars_set,
):
    """
    Reformulate a second-stage equality constraint that
    is independent of the state variables and subject
    to non-scenario-based uncertainty.

    If, after substitution of the decision rule expressions,
    the constraint expression is a polynomial (up to degree 2)
    in the uncertain parameters, then coefficient matching
    constraints are added. Note that in some (rare?) cases,
    coefficient matching constraints are restrictive.

    Otherwise, the constraint is cast to two second-stage
    inequality constraints, each of which is assigned a separation
    priority equal to ``DEFAULT_SEPARATION_PRIORITY``.

    The original constraint is removed from the model.

    Parameters
    ----------
    model_data : ModelData
        Model data object, with mostly preprocessed working model.
    config : ConfigDict
        PyROS solver settings.
    ss_eq_con : ConstraintData
        Second-stage equality constraint to be reformulated.
        Expected to be a member of
        ``model_data.working_model.second_stage.equality_cons``.
    ss_eq_con_index : hashable
        Index of the equality constraint in
        ``model_data.working_model.second_stage.equality_cons``.
    ss_var_id_to_dr_expr_map : dict
        Mapping from object IDs of second-stage variables to
        corresponding decision rule expressions.
    uncertain_param_id_to_temp_var_map : dict
        Mapping from object IDs of effective uncertain parameters
        to temporary placeholder variables.
    originally_unfixed_vars : list/ComponentSet of VarData
        Variables of the working model that were originally
        unfixed.
    all_dr_vars_set : ComponentSet
        A set of all the decision rule variables declared on
        ``model_data.working_model``.

    Returns
    -------
    robust_infeasible : bool
        True if robust infeasibility was detected through
        coefficient matching, False otherwise.
    """
    robust_infeasible = False

    working_model = model_data.working_model
    con_expr_after_dr_substitution = replace_expressions(
        expr=ss_eq_con.body - ss_eq_con.upper, substitution_map=ss_var_id_to_dr_expr_map
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
    # effective uncertain parameters only. thus, only the proxy
    # variables for the uncertain parameters are unfixed
    # during the analysis
    visitor = setup_quadratic_expression_visitor(wrt=originally_unfixed_vars)
    expr_repn = visitor.walk_expression(con_expr_after_all_substitutions)

    if expr_repn.nonlinear is not None:
        config.progress_logger.debug(
            f"Equality constraint {ss_eq_con.name!r} "
            "is state-variable independent, but cannot be written "
            "as a polynomial in the uncertain parameters with "
            "the currently available expression analyzers "
            "and selected decision rules "
            f"(decision_rule_order={config.decision_rule_order}). "
            "We are unable to write a coefficient matching reformulation "
            "of this constraint."
            "Recasting to two inequality constraints."
        )

        # keeping this constraint as an equality is not appropriate,
        # as it effectively constrains the uncertain parameters
        # in the separation problems, since the effective DOF
        # variables and DR variables are fixed.
        # hence, we reformulate to inequalities
        for bound_type in [BoundType.LOWER, BoundType.UPPER]:
            std_con_expr = create_bound_constraint_expr(
                expr=ss_eq_con.body, bound=ss_eq_con.upper, bound_type=bound_type
            )
            new_con_name = f"reform_{bound_type}_bound_from_{ss_eq_con_index}"
            working_model.second_stage.inequality_cons[new_con_name] = std_con_expr
            # no custom priorities specified
            model_data.separation_priority_order[new_con_name] = (
                model_data.separation_priority_order[ss_eq_con_index]
            )
    else:
        polynomial_repn_coeffs = (
            [expr_repn.constant]
            + list(expr_repn.linear.values())
            + (
                []
                if expr_repn.quadratic is None
                else list(expr_repn.quadratic.values())
            )
        )
        for coeff_idx, coeff_expr in enumerate(polynomial_repn_coeffs):
            # for robust satisfaction of the original equality
            # constraint, all polynomial coefficients must be
            # equal to zero. so for each coefficient,
            # we either check for trivial robust
            # feasibility/infeasibility, or add a constraint
            # restricting the coefficient expression to value 0
            if isinstance(coeff_expr, tuple(native_types)):
                # coefficient is a constant;
                # check value to determine
                # trivial feasibility/infeasibility
                robust_infeasible = not math.isclose(
                    a=coeff_expr,
                    b=0,
                    rel_tol=COEFF_MATCH_REL_TOL,
                    abs_tol=COEFF_MATCH_ABS_TOL,
                )
                if robust_infeasible:
                    config.progress_logger.info(
                        "PyROS has determined that the model is "
                        "robust infeasible. "
                        "One reason for this is that "
                        f"the equality constraint {ss_eq_con.name!r} "
                        "cannot be satisfied against all realizations "
                        "of uncertainty, "
                        "given the current partitioning into "
                        "first-stage, second-stage, and state variables. "
                        "Consider editing this constraint to reference some "
                        "(additional) second-stage and/or state variable(s)."
                    )

                    # robust infeasibility found;
                    # that is sufficient for termination of PyROS.
                    break

            else:
                # coefficient is variable-dependent.
                # add matching constraint
                new_con_name = f"coeff_matching_{ss_eq_con_index}_coeff_{coeff_idx}"
                vars_in_coeff_expr = ComponentSet(identify_variables(coeff_expr))
                has_expr_dr_vars = vars_in_coeff_expr & all_dr_vars_set
                indexed_con_to_update = (
                    working_model.first_stage.dr_dependent_equality_cons
                    if has_expr_dr_vars
                    else working_model.first_stage.dr_independent_equality_cons
                )
                indexed_con_to_update[new_con_name] = coeff_expr == 0
                new_con = indexed_con_to_update[new_con_name]
                working_model.first_stage.coefficient_matching_cons.append(new_con)

                dr_dependence_qual = (
                    f"DR variable-{'in' if has_expr_dr_vars else ''}dependent"
                )
                config.progress_logger.debug(
                    f"Derived from constraint {ss_eq_con.name!r} a "
                    f"{dr_dependence_qual} coefficient "
                    f"matching constraint named {new_con_name!r} "
                    "with expression: \n    "
                    f"{new_con.expr}."
                )

    del working_model.second_stage.equality_cons[ss_eq_con_index]

    return robust_infeasible


def reformulate_state_var_independent_eq_cons(model_data):
    """
    Reformulate second-stage equality constraints that are
    independent of the state variables.

    The reformulation of every such constraint is as follows:

    - If the uncertainty set is discrete, then the constraint,
      subject to each scenario in the set, is added to the
      first-stage equality constraints.
    - Otherwise:

      - If, after substitution of the decision rule expressions
        for the effective second-stage variables, the constraint
        expression is a polynomial (of degree up to 2) in the
        uncertain parameters, then an equality requiring that
        each coefficient be of value 0 is added to the first-stage
        equality constraints.
        In some cases, matching of the coefficients may lead to
        immediate detection of robust infeasibility.
      - Otherwise, the constraint is cast to two second-stage
        inequalities, each of which is assigned a separation
        priority of ``DEFAULT_SEPARATION_PRIORITY``.

    Parameters
    ----------
    model_data : model data object
        Main model data object.

    Returns
    -------
    robust_infeasible : bool
        True if model found to be robust infeasible,
        False otherwise.
    """
    config = model_data.config
    working_model = model_data.working_model
    ep = working_model.effective_var_partitioning

    effective_second_stage_var_set = ComponentSet(ep.second_stage_variables)
    effective_state_var_set = ComponentSet(ep.state_variables)
    all_vars_set = ComponentSet(working_model.all_variables)
    all_dr_vars_set = ComponentSet(
        generate_all_decision_rule_var_data_objects(working_model)
    )
    originally_unfixed_vars = [var for var in all_vars_set if not var.fixed]

    # we will need this to substitute DR expressions for
    # second-stage variables later
    ss_var_id_to_dr_expr_map = {
        id(ss_var): get_dr_expression(working_model, ss_var)
        for ss_var in effective_second_stage_var_set
    }

    # goal: examine constraint expressions in terms of the
    #       uncertain params. we will use standard repn to do this.
    # standard repn analyzes expressions in terms of Var components,
    # but the uncertain params are implemented as mutable Param objects
    # so we temporarily define Var components to be briefly substituted
    # for the uncertain parameters as the constraints are analyzed
    uncertain_params_set = ComponentSet(working_model.effective_uncertain_params)
    working_model.temp_param_vars = temp_param_vars = Var(
        range(len(uncertain_params_set)),
        initialize={
            idx: value(param) for idx, param in enumerate(uncertain_params_set)
        },
    )
    uncertain_param_to_temp_var_map = ComponentMap(
        (param, param_var)
        for param, param_var in zip(uncertain_params_set, temp_param_vars.values())
    )
    uncertain_param_id_to_temp_var_map = {
        id(param): var for param, var in uncertain_param_to_temp_var_map.items()
    }

    robust_infeasible = False

    # copy the items iterable,
    # as we will be modifying the constituents of the constraint
    # in place
    working_model.first_stage.coefficient_matching_cons = []
    for con_idx, con in list(working_model.second_stage.equality_cons.items()):
        vars_in_con = ComponentSet(identify_variables(con.expr))
        mutable_params_in_con = ComponentSet(identify_mutable_parameters(con.expr))

        second_stage_vars_in_con = vars_in_con & effective_second_stage_var_set
        state_vars_in_con = vars_in_con & effective_state_var_set
        uncertain_params_in_con = mutable_params_in_con & uncertain_params_set

        coefficient_matching_applicable = not state_vars_in_con and (
            uncertain_params_in_con or second_stage_vars_in_con
        )
        if coefficient_matching_applicable:
            if config.uncertainty_set.geometry.name == "DISCRETE_SCENARIOS":
                _reformulate_eq_con_scenario_uncertainty(
                    model_data=model_data,
                    ss_eq_con=con,
                    ss_eq_con_index=con_idx,
                    discrete_set=config.uncertainty_set,
                    ss_var_id_to_dr_expr_map=ss_var_id_to_dr_expr_map,
                    all_dr_vars_set=all_dr_vars_set,
                )
            else:
                robust_infeasible = _reformulate_eq_con_continuous_uncertainty(
                    model_data=model_data,
                    config=config,
                    ss_eq_con=con,
                    ss_eq_con_index=con_idx,
                    ss_var_id_to_dr_expr_map=ss_var_id_to_dr_expr_map,
                    uncertain_param_id_to_temp_var_map=(
                        uncertain_param_id_to_temp_var_map
                    ),
                    originally_unfixed_vars=originally_unfixed_vars,
                    all_dr_vars_set=all_dr_vars_set,
                )
                if robust_infeasible:
                    break
        del model_data.separation_priority_order[con_idx]

    # we no longer need these auxiliary components
    working_model.del_component(temp_param_vars)
    working_model.del_component(temp_param_vars.index_set())

    return robust_infeasible


def get_effective_uncertain_dimensions(model_data):
    """
    Determine the positional indices of the effective uncertain
    parameters, i.e., the uncertain parameters
    of a model that are not constrained to a single value
    by the uncertainty set constraints.

    Parameters
    ----------
    model_data : ModelData
        PyROS model data object.

    Returns
    -------
    list of int
        Positional indices of interest.
    """
    are_coordinates_fixed = model_data.config.uncertainty_set._is_coordinate_fixed(
        config=model_data.config
    )
    return [idx for idx, is_fixed in enumerate(are_coordinates_fixed) if not is_fixed]


def preprocess_model_data(model_data, user_var_partitioning):
    """
    Preprocess user inputs to modeling objects from which
    PyROS subproblems can be efficiently constructed.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    user_var_partitioning : VariablePartitioning
        User-based partitioning of the in-scope
        variables of the input model.

    Returns
    -------
    robust_infeasible : bool
        True if RO problem was found to be robust infeasible,
        False otherwise.
    """
    config = model_data.config
    setup_working_model(model_data, user_var_partitioning)

    config.progress_logger.debug(
        "Establishing the effective(ly) uncertain parameters..."
    )
    model_data.working_model.effective_uncertain_dimensions = (
        get_effective_uncertain_dimensions(model_data)
    )
    model_data.working_model.effective_uncertain_params = [
        model_data.working_model.uncertain_params[idx]
        for idx in model_data.working_model.effective_uncertain_dimensions
    ]

    # extract as many truly nonadjustable variables as possible
    # from the second-stage and state variables
    config.progress_logger.debug("Repartitioning variables by nonadjustability...")
    add_effective_var_partitioning(model_data)

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
    standardize_active_objective(model_data)

    # DR components are added only per effective second-stage variable
    config.progress_logger.debug("Adding decision rule components...")
    add_decision_rule_variables(model_data)
    add_decision_rule_constraints(model_data)

    # the epigraph and DR variables are also first-stage
    config.progress_logger.debug("Finalizing nonadjustable variables...")
    model_data.working_model.all_nonadjustable_variables = (
        get_all_nonadjustable_variables(model_data.working_model)
    )
    model_data.working_model.all_adjustable_variables = get_all_adjustable_variables(
        model_data.working_model
    )
    model_data.working_model.all_variables = (
        model_data.working_model.all_nonadjustable_variables
        + model_data.working_model.all_adjustable_variables
    )

    config.progress_logger.debug(
        "Reformulating state variable-independent second-stage equality constraints..."
    )
    robust_infeasible = reformulate_state_var_independent_eq_cons(model_data)

    # we are done looking for separation priorities
    for priority_sfx in model_data.separation_priority_suffix_finder.all_suffixes:
        priority_sfx.deactivate()

    return robust_infeasible


def log_model_statistics(model_data):
    """
    Log statistics for the preprocessed model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    """
    config = model_data.config
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
    num_dr_vars = len(list(generate_all_decision_rule_var_data_objects(working_model)))

    # uncertain parameters
    num_uncertain_params = len(working_model.uncertain_params)
    num_eff_uncertain_params = len(working_model.effective_uncertain_params)

    # constraints
    num_cons = len(list(working_model.component_data_objects(Constraint, active=True)))

    # # equality constraints
    num_eq_cons = (
        len(working_model.first_stage.dr_dependent_equality_cons)
        + len(working_model.first_stage.dr_independent_equality_cons)
        + len(working_model.second_stage.equality_cons)
        + len(working_model.second_stage.decision_rule_eqns)
    )
    num_first_stage_eq_cons = len(get_all_first_stage_eq_cons(working_model))
    num_coeff_matching_cons = len(working_model.first_stage.coefficient_matching_cons)
    num_other_first_stage_eqns = num_first_stage_eq_cons - num_coeff_matching_cons
    num_second_stage_eq_cons = len(working_model.second_stage.equality_cons)
    num_dr_eq_cons = len(working_model.second_stage.decision_rule_eqns)

    # # inequality constraints
    num_ineq_cons = len(working_model.first_stage.inequality_cons) + len(
        working_model.second_stage.inequality_cons
    )
    num_first_stage_ineq_cons = len(working_model.first_stage.inequality_cons)
    num_second_stage_ineq_cons = len(working_model.second_stage.inequality_cons)

    info_log_func = config.progress_logger.info

    IterationLogRecord.log_header_rule(info_log_func)
    info_log_func("Model Statistics:")

    info_log_func(f"  Number of variables : {num_vars}")
    info_log_func(f"    Epigraph variable : {num_epigraph_vars}")
    info_log_func(f"    First-stage variables : {num_first_stage_vars}")
    info_log_func(
        f"    Second-stage variables : {num_second_stage_vars} "
        f"({num_eff_second_stage_vars} adj.)"
    )
    info_log_func(
        f"    State variables : {num_state_vars} " f"({num_eff_state_vars} adj.)"
    )
    info_log_func(f"    Decision rule variables : {num_dr_vars}")

    info_log_func(
        f"  Number of uncertain parameters : {num_uncertain_params} "
        f"({num_eff_uncertain_params} eff.)"
    )

    info_log_func(f"  Number of constraints : {num_cons}")
    info_log_func(f"    Equality constraints : {num_eq_cons}")
    info_log_func(f"      Coefficient matching constraints : {num_coeff_matching_cons}")
    info_log_func(f"      Other first-stage equations : {num_other_first_stage_eqns}")
    info_log_func(f"      Second-stage equations : {num_second_stage_eq_cons}")
    info_log_func(f"      Decision rule equations : {num_dr_eq_cons}")
    info_log_func(f"    Inequality constraints : {num_ineq_cons}")
    info_log_func(f"      First-stage inequalities : {num_first_stage_ineq_cons}")
    info_log_func(f"      Second-stage inequalities : {num_second_stage_ineq_cons}")


def add_decision_rule_variables(model_data):
    """
    Add variables parameterizing the (polynomial)
    decision rules to the working model.

    Parameters
    ----------
    model_data : model data object
        Model data.

    Notes
    -----
    1. One set of decision rule variables is added for each
       effective second-stage variable.
    2. As an efficiency, no decision rule variables
       are added for the nonadjustable, user-defined second-stage
       variables, since the decision rules for such variables
       are necessarily nonstatic.
    """
    config = model_data.config
    effective_second_stage_vars = (
        model_data.working_model.effective_var_partitioning.second_stage_variables
    )
    model_data.working_model.first_stage.decision_rule_vars = decision_rule_vars = []

    # facilitate matching of effective second-stage vars to DR vars later
    model_data.working_model.eff_ss_var_to_dr_var_map = eff_ss_var_to_dr_var_map = (
        ComponentMap()
    )

    # since DR expression is a general polynomial in the uncertain
    # parameters, the exact number of DR variables
    # per effective second-stage variable
    # depends only on the DR order and uncertainty set dimension
    degree = config.decision_rule_order
    num_uncertain_params = len(model_data.working_model.effective_uncertain_params)
    num_dr_vars = sp.special.comb(
        N=num_uncertain_params + degree, k=degree, exact=True, repetition=False
    )

    for idx, eff_ss_var in enumerate(effective_second_stage_vars):
        indexed_dr_var = Var(
            range(num_dr_vars), initialize=0, bounds=(None, None), domain=Reals
        )
        model_data.working_model.first_stage.add_component(
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


def add_decision_rule_constraints(model_data):
    """
    Add decision rule equality constraints to the working model.

    Parameters
    ----------
    model_data : model data object
        Main model data object.
    """
    config = model_data.config
    effective_second_stage_vars = (
        model_data.working_model.effective_var_partitioning.second_stage_variables
    )
    indexed_dr_var_list = model_data.working_model.first_stage.decision_rule_vars
    uncertain_params = model_data.working_model.effective_uncertain_params
    degree = config.decision_rule_order

    model_data.working_model.second_stage.decision_rule_eqns = decision_rule_eqns = (
        Constraint(range(len(effective_second_stage_vars)))
    )

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
        decision_rule_eqns[idx] = dr_expression - eff_ss_var == 0
        eff_ss_var_to_dr_eqn_map[eff_ss_var] = decision_rule_eqns[idx]


def enforce_dr_degree(working_blk, config, degree):
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
    for indexed_dr_var in working_blk.first_stage.decision_rule_vars:
        for dr_var in indexed_dr_var.values():
            dr_var_degree = working_blk.dr_var_to_exponent_map[dr_var]
            if dr_var_degree > degree:
                dr_var.fix(0)
            else:
                dr_var.unfix()


def load_final_solution(model_data, master_soln, original_user_var_partitioning):
    """
    Load variable values from the master problem to the
    original model.

    Parameters
    ----------
    master_soln : MasterResults
        Master solution object, containing the master model.
    original_user_var_partitioning : VariablePartitioning
        User partitioning of the variables of the original
        model.
    """
    config = model_data.config
    if config.objective_focus == ObjectiveType.nominal:
        soln_master_blk = master_soln.master_model.scenarios[0, 0]
    elif config.objective_focus == ObjectiveType.worst_case:
        soln_master_blk = max(
            master_soln.master_model.scenarios.values(),
            key=lambda blk: value(blk.full_objective),
        )

    original_model_vars = (
        original_user_var_partitioning.first_stage_variables
        + original_user_var_partitioning.second_stage_variables
        + original_user_var_partitioning.state_variables
    )
    master_soln_vars = (
        soln_master_blk.user_var_partitioning.first_stage_variables
        + soln_master_blk.user_var_partitioning.second_stage_variables
        + soln_master_blk.user_var_partitioning.state_variables
    )
    for orig_var, master_blk_var in zip(original_model_vars, master_soln_vars):
        orig_var.set_value(master_blk_var.value, skip_validation=True)


def write_subproblem(model, fname, config):
    """
    Write/export a subproblem to one or more files.

    Parameters
    ----------
    model : ConcreteModel
        Subproblem to be written/exported.
    fname : str
        Base name of the file(s) to be written.
        Should not include any prefix directories.
    config : ConfigDict
        PyROS solver options.
        A file will be written for each format provided in
        ``config.subproblem_format_options``,
        and in the directory ``config.subproblem_file_directory``.
    """
    for fmt, io_options in config.subproblem_format_options.items():
        full_filename = os.path.join(config.subproblem_file_directory, f"{fname}.{fmt}")
        model.write(filename=full_filename, format=fmt, io_options=io_options)
        config.progress_logger.warning(
            f"For debugging, subproblem has been written to the file {full_filename!r}."
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
    except (ApplicationError, InvalidValueError):
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
        Number of second-stage constraints found to be violated
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
        Maximum scaled violation of any second-stage constraint
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
        Number of second-stage constraints found to be violated
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
        Maximum scaled violation of any second-stage constraint
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


def copy_docstring(source_func):
    """
    Create a decorator which copies docstring of a callable
    `source_func` to a target callable passed to the decorator.

    Returns
    -------
    decorator_doc : callable
        Decorator of interest.
    """

    def decorator_doc(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = source_func.__doc__
        return wrapper

    return decorator_doc
