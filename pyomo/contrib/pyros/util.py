'''
Utility functions for the PyROS solver
'''
import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet
from pyomo.common.modeling import unique_component_name
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
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
import pyomo.core.expr as EXPR
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
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
from pprint import pprint
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
    Adjust solver max time setting based on current PyROS elapsed
    time.

    Parameters
    ----------
    timing_data_obj : Bunch
        PyROS timekeeper.
    solver : solver type
        Solver for which to adjust the max time setting.
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
        after a generic interface to the time limit setting
        is introduced.
    (2) For IPOPT, and probably also BARON, the CPU time limit
        rather than the wallclock time limit, is adjusted, as
        no interface to wallclock limit available.
        For this reason, extra 30s is added to time remaining
        for subsolver time limit.
        (The extra 30s is large enough to ensure solver
        elapsed time is not beneath elapsed time - user time limit,
        but not so large as to overshoot the user-specified time limit
        by an inordinate margin.)
    """
    if config.time_limit is not None:
        time_remaining = config.time_limit - get_main_elapsed_time(timing_data_obj)
        if isinstance(solver, type(SolverFactory("gams", solver_io="shell"))):
            original_max_time_setting = solver.options["add_options"]
            custom_setting_present = "add_options" in solver.options

            # adjust GAMS solver time
            reslim_str = f"option reslim={max(30, 30 + time_remaining)};"
            if isinstance(solver.options["add_options"], list):
                solver.options["add_options"].append(reslim_str)
            else:
                solver.options["add_options"] = [reslim_str]
        else:
            # determine name of option to adjust
            if isinstance(solver, SolverFactory.get_class("baron")):
                options_key = "MaxTime"
            elif isinstance(solver, SolverFactory.get_class("ipopt")):
                options_key = "max_cpu_time"
            else:
                options_key = None

            if options_key is not None:
                custom_setting_present = options_key in solver.options
                original_max_time_setting = solver.options[options_key]

                # ensure positive value assigned to avoid application error
                solver.options[options_key] = max(30, 30 + time_remaining)
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
            options_key = "max_cpu_time"
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
                # remove the max time specification introduced.
                # All lines are needed here to completely remove the option
                # from access through getattr and dictionary reference.
                delattr(solver.options, options_key)
                if options_key in solver.options.keys():
                    del solver.options[options_key]


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


def a_logger(str_or_logger):
    """
    Standardize a string or logger object to a logger object.

    Parameters
    ----------
    str_or_logger : str or logging.Logger
        String or logger object to normalize.

    Returns
    -------
    logging.Logger
        If `str_or_logger` is of type `logging.Logger`,then
        `str_or_logger` is returned.
        Otherwise, ``logging.getLogger(str_or_logger)``
        is returned. In the event `str_or_logger` is
        the name of the default PyROS logger, the logger level
        is set to `logging.INFO`, and a `PreformattedLogger`
        instance is returned in lieu of a standard `Logger`
        instance.
    """
    if isinstance(str_or_logger, logging.Logger):
        return logging.getLogger(str_or_logger.name)
    else:
        return logging.getLogger(str_or_logger)


def ValidEnum(enum_class):
    '''
    Python 3 dependent format string
    '''

    def fcn(obj):
        if obj not in enum_class:
            raise ValueError(
                "Expected an {0} object, "
                "instead received {1}".format(
                    enum_class.__name__, obj.__class__.__name__
                )
            )
        return obj

    return fcn


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


def model_is_valid(model):
    """
    Assess whether model is valid on basis of the number of active
    Objectives. A valid model must contain exactly one active Objective.
    """
    return len(list(model.component_data_objects(Objective, active=True))) == 1


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


def validate_uncertainty_set(config):
    '''
    Confirm expression output from uncertainty set function references all q in q.
    Typecheck the uncertainty_set.q is Params referenced inside of m.
    Give warning that the nominal point (default value in the model) is not in the specified uncertainty set.
    :param config: solver config
    '''
    # === Check that q in UncertaintySet object constraint expression is referencing q in model.uncertain_params
    uncertain_params = config.uncertain_params

    # === Non-zero number of uncertain parameters
    if len(uncertain_params) == 0:
        raise AttributeError(
            "Must provide uncertain params, uncertain_params list length is 0."
        )
    # === No duplicate parameters
    if len(uncertain_params) != len(ComponentSet(uncertain_params)):
        raise AttributeError("No duplicates allowed for uncertain param objects.")
    # === Ensure nominal point is in the set
    if not config.uncertainty_set.point_in_set(
        point=config.nominal_uncertain_param_vals
    ):
        raise AttributeError(
            "Nominal point for uncertain parameters must be in the uncertainty set."
        )
    # === Check set validity via boundedness and non-emptiness
    if not config.uncertainty_set.is_valid(config=config):
        raise AttributeError(
            "Invalid uncertainty set detected. Check the uncertainty set object to "
            "ensure non-emptiness and boundedness."
        )

    return


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


def validate_kwarg_inputs(model, config):
    '''
    Confirm kwarg inputs satisfy PyROS requirements.
    :param model: the deterministic model
    :param config: the config for this PyROS instance
    :return:
    '''

    # === Check if model is ConcreteModel object
    if not isinstance(model, ConcreteModel):
        raise ValueError("Model passed to PyROS solver must be a ConcreteModel object.")

    first_stage_variables = config.first_stage_variables
    second_stage_variables = config.second_stage_variables
    uncertain_params = config.uncertain_params

    if not config.first_stage_variables and not config.second_stage_variables:
        # Must have non-zero DOF
        raise ValueError(
            "first_stage_variables and "
            "second_stage_variables cannot both be empty lists."
        )

    if ComponentSet(first_stage_variables) != ComponentSet(
        config.first_stage_variables
    ):
        raise ValueError(
            "All elements in first_stage_variables must be Var members of the model object."
        )

    if ComponentSet(second_stage_variables) != ComponentSet(
        config.second_stage_variables
    ):
        raise ValueError(
            "All elements in second_stage_variables must be Var members of the model object."
        )

    if any(
        v in ComponentSet(second_stage_variables)
        for v in ComponentSet(first_stage_variables)
    ):
        raise ValueError(
            "No common elements allowed between first_stage_variables and second_stage_variables."
        )

    if ComponentSet(uncertain_params) != ComponentSet(config.uncertain_params):
        raise ValueError(
            "uncertain_params must be mutable Param members of the model object."
        )

    if not config.uncertainty_set:
        raise ValueError(
            "An UncertaintySet object must be provided to the PyROS solver."
        )

    non_mutable_params = []
    for p in config.uncertain_params:
        if not (
            not p.is_constant() and p.is_fixed() and not p.is_potentially_variable()
        ):
            non_mutable_params.append(p)
        if non_mutable_params:
            raise ValueError(
                "Param objects which are uncertain must have attribute mutable=True. "
                "Offending Params: %s" % [p.name for p in non_mutable_params]
            )

    # === Solvers provided check
    if not config.local_solver or not config.global_solver:
        raise ValueError(
            "User must designate both a local and global optimization solver via the local_solver"
            " and global_solver options."
        )

    if config.bypass_local_separation and config.bypass_global_separation:
        raise ValueError(
            "User cannot simultaneously enable options "
            "'bypass_local_separation' and "
            "'bypass_global_separation'."
        )

    # === Degrees of freedom provided check
    if len(config.first_stage_variables) + len(config.second_stage_variables) == 0:
        raise ValueError(
            "User must designate at least one first- and/or second-stage variable."
        )

    # === Uncertain params provided check
    if len(config.uncertain_params) == 0:
        raise ValueError("User must designate at least one uncertain parameter.")

    return


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
    '''
    Function to add decision rule (DR) variables to the working model. DR variables become first-stage design
    variables which do not get copied at each iteration. Currently support static_approx (no DR), affine DR,
    and quadratic DR.
    :param model_data: the data container for the working model
    :param config: the config block
    :return:
    '''
    second_stage_variables = model_data.working_model.util.second_stage_variables
    first_stage_variables = model_data.working_model.util.first_stage_variables
    uncertain_params = model_data.working_model.util.uncertain_params
    decision_rule_vars = []
    degree = config.decision_rule_order
    bounds = (None, None)
    if degree == 0:
        for i in range(len(second_stage_variables)):
            model_data.working_model.add_component(
                "decision_rule_var_" + str(i),
                Var(
                    initialize=value(second_stage_variables[i], exception=False),
                    bounds=bounds,
                    domain=Reals,
                ),
            )
            first_stage_variables.extend(
                getattr(
                    model_data.working_model, "decision_rule_var_" + str(i)
                ).values()
            )
            decision_rule_vars.append(
                getattr(model_data.working_model, "decision_rule_var_" + str(i))
            )
    elif degree == 1:
        for i in range(len(second_stage_variables)):
            index_set = list(range(len(uncertain_params) + 1))
            model_data.working_model.add_component(
                "decision_rule_var_" + str(i),
                Var(index_set, initialize=0, bounds=bounds, domain=Reals),
            )
            # === For affine drs, the [0]th constant term is initialized to the control variable values, all other terms are initialized to 0
            getattr(model_data.working_model, "decision_rule_var_" + str(i))[
                0
            ].set_value(
                value(second_stage_variables[i], exception=False), skip_validation=True
            )
            first_stage_variables.extend(
                list(
                    getattr(
                        model_data.working_model, "decision_rule_var_" + str(i)
                    ).values()
                )
            )
            decision_rule_vars.append(
                getattr(model_data.working_model, "decision_rule_var_" + str(i))
            )
    elif degree == 2 or degree == 3 or degree == 4:
        for i in range(len(second_stage_variables)):
            num_vars = int(sp.special.comb(N=len(uncertain_params) + degree, k=degree))
            dict_init = {}
            for r in range(num_vars):
                if r == 0:
                    dict_init.update(
                        {r: value(second_stage_variables[i], exception=False)}
                    )
                else:
                    dict_init.update({r: 0})
            model_data.working_model.add_component(
                "decision_rule_var_" + str(i),
                Var(
                    list(range(num_vars)),
                    initialize=dict_init,
                    bounds=bounds,
                    domain=Reals,
                ),
            )
            first_stage_variables.extend(
                list(
                    getattr(
                        model_data.working_model, "decision_rule_var_" + str(i)
                    ).values()
                )
            )
            decision_rule_vars.append(
                getattr(model_data.working_model, "decision_rule_var_" + str(i))
            )
    else:
        raise ValueError(
            "Decision rule order "
            + str(config.decision_rule_order)
            + " is not yet supported. PyROS supports polynomials of degree 0 (static approximation), 1, 2."
        )
    model_data.working_model.util.decision_rule_vars = decision_rule_vars


def partition_powers(n, v):
    """Partition a total degree n across v variables

    This is an implementation of the "stars and bars" algorithm from
    combinatorial mathematics.

    This partitions a "total integer degree" of n across v variables
    such that each variable gets an integer degree >= 0.  You can think
    of this as dividing a set of n+v things into v groupings, with the
    power for each v_i being 1 less than the number of things in the
    i'th group (because the v is part of the group).  It is therefore
    sufficient to just get the v-1 starting points chosen from a list of
    indices n+v long (the first starting point is fixed to be 0).

    """
    for starts in it.combinations(range(1, n + v), v - 1):
        # add the initial starting point to the beginning and the total
        # number of objects (degree counters and variables) to the end
        # of the list.  The degree for each variable is 1 less than the
        # difference of sequential starting points (to account for the
        # variable itself)
        starts = (0,) + starts + (n + v,)
        yield [starts[i + 1] - starts[i] - 1 for i in range(v)]


def sort_partitioned_powers(powers_list):
    powers_list = sorted(powers_list, reverse=True)
    powers_list = sorted(powers_list, key=lambda elem: max(elem))
    return powers_list


def add_decision_rule_constraints(model_data, config):
    '''
    Function to add the defining Constraint relationships for the decision rules to the working model.
    :param model_data: model data container object
    :param config: the config object
    :return:
    '''

    second_stage_variables = model_data.working_model.util.second_stage_variables
    uncertain_params = model_data.working_model.util.uncertain_params
    decision_rule_eqns = []
    degree = config.decision_rule_order
    if degree == 0:
        for i in range(len(second_stage_variables)):
            model_data.working_model.add_component(
                "decision_rule_eqn_" + str(i),
                Constraint(
                    expr=getattr(
                        model_data.working_model, "decision_rule_var_" + str(i)
                    )
                    == second_stage_variables[i]
                ),
            )
            decision_rule_eqns.append(
                getattr(model_data.working_model, "decision_rule_eqn_" + str(i))
            )
    elif degree == 1:
        for i in range(len(second_stage_variables)):
            expr = 0
            for j in range(
                len(getattr(model_data.working_model, "decision_rule_var_" + str(i)))
            ):
                if j == 0:
                    expr += getattr(
                        model_data.working_model, "decision_rule_var_" + str(i)
                    )[j]
                else:
                    expr += (
                        getattr(
                            model_data.working_model, "decision_rule_var_" + str(i)
                        )[j]
                        * uncertain_params[j - 1]
                    )
            model_data.working_model.add_component(
                "decision_rule_eqn_" + str(i),
                Constraint(expr=expr == second_stage_variables[i]),
            )
            decision_rule_eqns.append(
                getattr(model_data.working_model, "decision_rule_eqn_" + str(i))
            )
    elif degree >= 2:
        # Using bars and stars groupings of variable powers, construct x1^a * .... * xn^b terms for all c <= a+...+b = degree
        all_powers = []
        for n in range(1, degree + 1):
            all_powers.append(
                sort_partitioned_powers(
                    list(partition_powers(n, len(uncertain_params)))
                )
            )
        for i in range(len(second_stage_variables)):
            Z = list(
                z
                for z in getattr(
                    model_data.working_model, "decision_rule_var_" + str(i)
                ).values()
            )
            e = Z.pop(0)
            for degree_param_powers in all_powers:
                for param_powers in degree_param_powers:
                    product = 1
                    for idx, power in enumerate(param_powers):
                        if power == 0:
                            pass
                        else:
                            product = product * uncertain_params[idx] ** power
                    e += Z.pop(0) * product
            model_data.working_model.add_component(
                "decision_rule_eqn_" + str(i),
                Constraint(expr=e == second_stage_variables[i]),
            )
            decision_rule_eqns.append(
                getattr(model_data.working_model, "decision_rule_eqn_" + str(i))
            )
            if len(Z) != 0:
                raise RuntimeError(
                    "Construction of the decision rule functions did not work correctly! "
                    "Did not use all coefficient terms."
                )
    model_data.working_model.util.decision_rule_eqns = decision_rule_eqns


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
