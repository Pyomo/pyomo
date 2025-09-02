"""
Interfaces for managing PyROS solver options.
"""

import logging

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    In,
    IsInstance,
    NonNegativeFloat,
    InEnum,
    Path,
)
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import ApplicationError, PyomoException
from pyomo.core.base import Var, VarData
from pyomo.core.base.param import Param, ParamData
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import (
    ObjectiveType,
    setup_pyros_logger,
    standardize_component_data,
)
from pyomo.contrib.pyros.uncertainty_sets import UncertaintySet


default_pyros_solver_logger = setup_pyros_logger()


def logger_domain(obj):
    """
    Domain validator for logger-type arguments.

    This admits any object of type ``logging.Logger``,
    or which can be cast to ``logging.Logger``.
    """
    if isinstance(obj, logging.Logger):
        return obj
    else:
        return logging.getLogger(obj)


logger_domain.domain_name = "None, str or logging.Logger"


def positive_int_or_minus_one(obj):
    """
    Domain validator for objects castable to a strictly
    positive int or -1.
    """
    ans = int(obj)
    if ans != float(obj) or (ans <= 0 and ans != -1):
        raise ValueError(f"Expected positive int or -1, but received value {obj!r}")
    return ans


positive_int_or_minus_one.domain_name = "positive int or -1"


def _deprecated_separation_priority_order(obj):
    """
    Domain validator for argument `separation_priority_order`.

    As this argument has been deprecated, a deprecation warning
    is issued through a WARNING-level logger message if
    the argument is cast to a nonempty dict.

    Parameters
    ----------
    obj : object
        Argument value.

    Returns
    -------
    separation_priority_order : dict
        Argument value, cast to a dict.
    """
    separation_priority_order = dict(obj)
    if separation_priority_order:
        deprecation_warning(
            "The argument 'separation_priority_order' is deprecated. "
            "Consider specifying separation priorities by declaring, on your "
            "model, Suffix components with local name `pyros_separation_priority`.",
            version="6.9.3",
        )
    return separation_priority_order


_deprecated_separation_priority_order.domain_name = dict.__name__


def uncertain_param_validator(uncertain_obj):
    """
    Check that a component object modeling an
    uncertain parameter in PyROS is appropriately constructed,
    initialized, and/or mutable, where applicable.

    Parameters
    ----------
    uncertain_obj : Param or Var
        Object on which to perform checks.

    Raises
    ------
    ValueError
        If the length of the component (data) object does not
        match that of its index set, or the object is a Param
        with attribute `mutable=False`.
    """
    if len(uncertain_obj) != len(uncertain_obj.index_set()):
        raise ValueError(
            f"Length of {type(uncertain_obj).__name__} object with "
            f"name {uncertain_obj.name!r} is {len(uncertain_obj)}, "
            "and does not match that of its index set, "
            f"which is of length {len(uncertain_obj.index_set())}. "
            "Check that the component has been properly constructed, "
            "and all entries have been initialized. "
        )
    if uncertain_obj.ctype is Param and not uncertain_obj.mutable:
        raise ValueError(
            f"{type(uncertain_obj).__name__} object with name {uncertain_obj.name!r} "
            "is immutable."
        )


def uncertain_param_data_validator(uncertain_obj):
    """
    Validator for component data object specified as an
    uncertain parameter.

    Parameters
    ----------
    uncertain_obj : ParamData or VarData
        Object on which to perform checks.

    Raises
    ------
    ValueError
        If `uncertain_obj` is a VarData object
        that is not fixed explicitly via VarData.fixed
        or implicitly via bounds.
    """
    if isinstance(uncertain_obj, VarData):
        is_fixed_var = uncertain_obj.fixed or (
            uncertain_obj.lower is uncertain_obj.upper
            and uncertain_obj.lower is not None
        )
        if not is_fixed_var:
            raise ValueError(
                f"{type(uncertain_obj).__name__} object with name "
                f"{uncertain_obj.name!r} is not fixed."
            )


class InputDataStandardizer(object):
    """
    Domain validator for an object that is castable to
    a list of Pyomo component data objects.

    Parameters
    ----------
    ctype : type or tuple of type
        Valid Pyomo component type(s),
        such as Component, Var or Param.
    cdatatype : type or tuple of type
        Valid Pyomo component data type(s), such as
        ComponentData, VarData, or ParamData.
    ctype_validator : callable, optional
        Validator function for objects of type `ctype`.
    cdatatype_validator : callable, optional
        Validator function for objects of type `cdatatype`.
    allow_repeats : bool, optional
        True to allow duplicate component data object
        entries in final list to which argument is cast,
        False otherwise.

    Attributes
    ----------
    ctype : type or tuple of type
    cdatatype : type or tuple of type
    ctype_validator : callable or None
    cdatatype_validator : callable or None
    allow_repeats : bool
    """

    def __init__(
        self,
        ctype,
        cdatatype,
        ctype_validator=None,
        cdatatype_validator=None,
        allow_repeats=False,
    ):
        """Initialize self (see class docstring)."""
        self.ctype = ctype
        self.cdatatype = cdatatype
        self.ctype_validator = ctype_validator
        self.cdatatype_validator = cdatatype_validator
        self.allow_repeats = allow_repeats

    def __call__(self, obj, from_iterable=None, allow_repeats=None):
        """
        Cast object to a flat list of Pyomo component data type
        entries.

        Parameters
        ----------
        obj : object
            Object to be cast.
        from_iterable : Iterable or None, optional
            Iterable from which `obj` obtained, if any.
        allow_repeats : bool or None, optional
            True if list can contain repeated entries,
            False otherwise.

        Returns
        -------
        list of ComponentData
            Each entry is an instance of ``self.cdatatype``.
        """
        return standardize_component_data(
            obj=obj,
            valid_ctype=self.ctype,
            valid_cdatatype=self.cdatatype,
            ctype_validator=self.ctype_validator,
            cdatatype_validator=self.cdatatype_validator,
            allow_repeats=allow_repeats,
            from_iterable=from_iterable,
        )

    def domain_name(self):
        """Return str briefly describing domain encompassed by self."""
        ctypes_tup = (self.ctype,) if isinstance(self.ctype, type) else self.ctype
        cdtypes_tup = (
            (self.cdatatype,) if isinstance(self.cdatatype, type) else self.cdatatype
        )
        alltypes_desc = ", ".join(vtype.__name__ for vtype in ctypes_tup + cdtypes_tup)
        return f"(iterable of) {alltypes_desc}"


class SolverNotResolvable(PyomoException):
    """
    Exception type for failure to cast an object to a Pyomo solver.
    """


class SolverResolvable(object):
    """
    Callable for casting an object (such as a str)
    to a Pyomo solver.

    Parameters
    ----------
    require_available : bool, optional
        True if `available()` method of a standardized solver
        object obtained through `self` must return `True`,
        False otherwise.
    solver_desc : str, optional
        Descriptor for the solver obtained through `self`,
        such as 'local solver'
        or 'global solver'. This argument is used
        for constructing error/exception messages.

    Attributes
    ----------
    require_available
    solver_desc
    """

    def __init__(self, require_available=True, solver_desc="solver"):
        """Initialize self (see class docstring)."""
        self.require_available = require_available
        self.solver_desc = solver_desc

    @staticmethod
    def is_solver_type(obj):
        """
        Return True if object is considered a Pyomo solver,
        False otherwise.

        An object is considered a Pyomo solver provided that
        it has callable attributes named 'solve' and
        'available'.
        """
        return callable(getattr(obj, "solve", None)) and callable(
            getattr(obj, "available", None)
        )

    def __call__(self, obj, require_available=None, solver_desc=None):
        """
        Cast object to a Pyomo solver.

        If `obj` is a string, then ``SolverFactory(obj.lower())``
        is returned. If `obj` is a Pyomo solver type, then
        `obj` is returned.

        Parameters
        ----------
        obj : object
            Object to be cast to Pyomo solver type.
        require_available : bool or None, optional
            True if `available()` method of the resolved solver
            object must return True, False otherwise.
            If `None` is passed, then ``self.require_available``
            is used.
        solver_desc : str or None, optional
            Brief description of the solver, such as 'local solver'
            or 'backup global solver'. This argument is used
            for constructing error/exception messages.
            If `None` is passed, then ``self.solver_desc``
            is used.

        Returns
        -------
        Solver
            Pyomo solver.

        Raises
        ------
        SolverNotResolvable
            If `obj` cannot be cast to a Pyomo solver because
            it is neither a str nor a Pyomo solver type.
        ApplicationError
            In event that solver is not available, the
            method `available(exception_flag=True)` of the
            solver to which `obj` is cast should raise an
            exception of this type. The present method
            will also emit a more detailed error message
            through the default PyROS logger.
        """
        # resort to defaults if necessary
        if require_available is None:
            require_available = self.require_available
        if solver_desc is None:
            solver_desc = self.solver_desc

        # perform casting
        if isinstance(obj, str):
            solver = SolverFactory(obj.lower())
        elif self.is_solver_type(obj):
            solver = obj
        else:
            raise SolverNotResolvable(
                f"Cannot cast object `{obj!r}` to a Pyomo optimizer for use as "
                f"{solver_desc}, as the object is neither a str nor a "
                f"Pyomo Solver type (got type {type(obj).__name__})."
            )

        # availability check, if so desired
        if require_available:
            try:
                solver.available(exception_flag=True)
            except ApplicationError:
                default_pyros_solver_logger.exception(
                    f"Output of `available()` method for {solver_desc} "
                    f"with repr {solver!r} resolved from object {obj} "
                    "is not `True`. "
                    "Check solver and any required dependencies "
                    "have been set up properly."
                )
                raise

        return solver

    def domain_name(self):
        """Return str briefly describing domain encompassed by self."""
        return "str or Solver"


class SolverIterable(object):
    """
    Callable for casting an iterable (such as a list of strs)
    to a list of Pyomo solvers.

    Parameters
    ----------
    require_available : bool, optional
        True if `available()` method of a standardized solver
        object obtained through `self` must return `True`,
        False otherwise.
    filter_by_availability : bool, optional
        True to remove standardized solvers for which `available()`
        does not return True, False otherwise.
    solver_desc : str, optional
        Descriptor for the solver obtained through `self`,
        such as 'backup local solver'
        or 'backup global solver'.
    """

    def __init__(
        self, require_available=True, filter_by_availability=True, solver_desc="solver"
    ):
        """Initialize self (see class docstring)."""
        self.require_available = require_available
        self.filter_by_availability = filter_by_availability
        self.solver_desc = solver_desc

    def __call__(
        self, obj, require_available=None, filter_by_availability=None, solver_desc=None
    ):
        """
        Cast iterable object to a list of Pyomo solver objects.

        Parameters
        ----------
        obj : str, Solver, or Iterable of str/Solver
            Object of interest.
        require_available : bool or None, optional
            True if `available()` method of each solver
            object must return True, False otherwise.
            If `None` is passed, then ``self.require_available``
            is used.
        solver_desc : str or None, optional
            Descriptor for the solver, such as 'backup local solver'
            or 'backup global solver'. This argument is used
            for constructing error/exception messages.
            If `None` is passed, then ``self.solver_desc``
            is used.

        Returns
        -------
        solvers : list of solver type
            List of solver objects to which obj is cast.

        Raises
        ------
        TypeError
            If `obj` is a str.
        """
        if require_available is None:
            require_available = self.require_available
        if filter_by_availability is None:
            filter_by_availability = self.filter_by_availability
        if solver_desc is None:
            solver_desc = self.solver_desc

        solver_resolve_func = SolverResolvable()

        if isinstance(obj, str) or solver_resolve_func.is_solver_type(obj):
            # single solver resolvable is cast to singleton list.
            # perform explicit check for str, otherwise this method
            # would attempt to resolve each character.
            obj_as_list = [obj]
        else:
            obj_as_list = list(obj)

        solvers = []
        for idx, val in enumerate(obj_as_list):
            solver_desc_str = f"{solver_desc} " f"(index {idx})"
            opt = solver_resolve_func(
                obj=val,
                require_available=require_available,
                solver_desc=solver_desc_str,
            )
            if filter_by_availability and not opt.available(exception_flag=False):
                default_pyros_solver_logger.warning(
                    f"Output of `available()` method for solver object {opt} "
                    f"resolved from object {val} of sequence {obj_as_list} "
                    f"to be used as {self.solver_desc} "
                    "is not `True`. "
                    "Removing from list of standardized solvers."
                )
            else:
                solvers.append(opt)

        return solvers

    def domain_name(self):
        """Return str briefly describing domain encompassed by self."""
        return "str, solver type, or Iterable of str/solver type"


def pyros_config():
    CONFIG = ConfigDict('PyROS')

    # ================================================
    # === Options common to all solvers
    # ================================================
    CONFIG.declare(
        'time_limit',
        ConfigValue(
            default=None,
            domain=NonNegativeFloat,
            doc=(
                """
                Wall time limit for the execution of the PyROS solver
                in seconds (including time spent by subsolvers).
                If `None` is provided, then no time limit is enforced.
                """
            ),
        ),
    )
    CONFIG.declare(
        'keepfiles',
        ConfigValue(
            default=False,
            domain=bool,
            description=(
                """
                Export subproblems with a non-acceptable termination status
                for debugging purposes.
                If True is provided, then the argument
                `subproblem_file_directory` must also be specified.
                """
            ),
        ),
    )
    CONFIG.declare(
        'tee',
        ConfigValue(
            default=False,
            domain=bool,
            description="Output subordinate solver logs for all subproblems.",
        ),
    )
    CONFIG.declare(
        'load_solution',
        ConfigValue(
            default=True,
            domain=bool,
            description=(
                """
                Load final solution(s) found by PyROS to the deterministic
                model provided.
                """
            ),
        ),
    )
    CONFIG.declare(
        'symbolic_solver_labels',
        ConfigValue(
            default=False,
            domain=bool,
            description=(
                """
                True to ensure the component names given to the
                subordinate solvers for every subproblem reflect
                the names of the corresponding Pyomo modeling components,
                False otherwise.
                """
            ),
        ),
    )

    # ================================================
    # === Required User Inputs
    # ================================================
    CONFIG.declare(
        "first_stage_variables",
        ConfigValue(
            default=[],
            domain=InputDataStandardizer(Var, VarData, allow_repeats=False),
            description="First-stage (or design) variables.",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "second_stage_variables",
        ConfigValue(
            default=[],
            domain=InputDataStandardizer(Var, VarData, allow_repeats=False),
            description="Second-stage (or control) variables.",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "uncertain_params",
        ConfigValue(
            default=[],
            domain=InputDataStandardizer(
                ctype=(Param, Var),
                cdatatype=(ParamData, VarData),
                ctype_validator=uncertain_param_validator,
                cdatatype_validator=uncertain_param_data_validator,
                allow_repeats=False,
            ),
            description=(
                """
                Uncertain model parameters.
                Of every constituent `Param` object,
                the `mutable` attribute must be set to True.
                All constituent `Var`/`VarData` objects should be
                fixed.
                """
            ),
            visibility=1,
        ),
    )
    CONFIG.declare(
        "uncertainty_set",
        ConfigValue(
            default=None,
            domain=IsInstance(UncertaintySet),
            description=(
                """
                Uncertainty set against which the
                final solution(s) returned by PyROS should be certified
                to be robust.
                """
            ),
            visibility=1,
        ),
    )
    CONFIG.declare(
        "local_solver",
        ConfigValue(
            default=None,
            domain=SolverResolvable(solver_desc="local solver", require_available=True),
            description="Subordinate local NLP solver.",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "global_solver",
        ConfigValue(
            default=None,
            domain=SolverResolvable(
                solver_desc="global solver", require_available=True
            ),
            description="Subordinate global NLP solver.",
            visibility=1,
        ),
    )
    # ================================================
    # === Optional User Inputs
    # ================================================
    CONFIG.declare(
        "objective_focus",
        ConfigValue(
            default=ObjectiveType.nominal,
            domain=InEnum(ObjectiveType),
            description=(
                """
                Choice of objective focus to optimize in the master problems.
                Choices are: `ObjectiveType.worst_case`,
                `ObjectiveType.nominal`.
                """
            ),
            doc=(
                """
                Objective focus for the master problems:

                - `ObjectiveType.nominal`:
                  Optimize the objective function subject to the nominal
                  uncertain parameter realization.
                - `ObjectiveType.worst_case`:
                  Optimize the objective function subject to the worst-case
                  uncertain parameter realization.

                By default, `ObjectiveType.nominal` is chosen.

                A worst-case objective focus is required for certification
                of robust optimality of the final solution(s) returned
                by PyROS.
                If a nominal objective focus is chosen, then only robust
                feasibility is guaranteed.
                """
            ),
        ),
    )
    CONFIG.declare(
        "nominal_uncertain_param_vals",
        ConfigValue(
            default=[],
            domain=list,
            doc=(
                """
                Nominal uncertain parameter realization.
                Entries should be provided in an order consistent with the
                entries of the argument `uncertain_params`.
                If an empty list is provided, then the values of the `Param`
                objects specified through `uncertain_params` are chosen.
                """
            ),
        ),
    )
    CONFIG.declare(
        "decision_rule_order",
        ConfigValue(
            default=0,
            domain=In([0, 1, 2]),
            description=(
                """
                Order (or degree) of the polynomial decision rule functions
                used for approximating the adjustability of the second stage
                variables with respect to the uncertain parameters.
                """
            ),
            doc=(
                """
                Order (or degree) of the polynomial decision rule functions
                for approximating the adjustability of the second stage
                variables with respect to the uncertain parameters.

                Choices are:

                - 0: static recourse
                - 1: affine recourse
                - 2: quadratic recourse
                """
            ),
        ),
    )
    CONFIG.declare(
        "solve_master_globally",
        ConfigValue(
            default=False,
            domain=bool,
            doc=(
                """
                True to solve all master problems with the subordinate
                global solver, False to solve all master problems with
                the subordinate local solver.
                Along with a worst-case objective focus
                (see argument `objective_focus`),
                solving the master problems to global optimality is required
                for certification
                of robust optimality of the final solution(s) returned
                by PyROS. Otherwise, only robust feasibility is guaranteed.
                """
            ),
        ),
    )
    CONFIG.declare(
        "max_iter",
        ConfigValue(
            default=-1,
            domain=positive_int_or_minus_one,
            description=(
                """
                Iteration limit. If -1 is provided, then no iteration
                limit is enforced.
                """
            ),
        ),
    )
    CONFIG.declare(
        "robust_feasibility_tolerance",
        ConfigValue(
            default=1e-4,
            domain=NonNegativeFloat,
            description=(
                """
                Relative tolerance for assessing maximal inequality
                constraint violations during the GRCS separation step.
                """
            ),
        ),
    )
    CONFIG.declare(
        "separation_priority_order",
        ConfigValue(
            default={},
            domain=_deprecated_separation_priority_order,
            doc=(
                """
                (DEPRECATED)
                A dict-like object, each entry of which
                maps the full name of a model ``Var`` or ``Constraint``
                component to a value specifying the separation priority
                for all constraints derived from the component.
                A separation priority can be a numeric value or None.
                A higher numeric value indicates a higher priority.
                For all constraints, the default priority is 0.
                (Inequality and equality) constraints with a
                priority of None are excluded from
                the separation problems and enforced subject to only
                the nominal uncertain parameter realization in the master
                problems.
                Separation problems corresponding to inequality
                constraints with numeric priorities are grouped by
                priority. In every iteration, the groups are traversed
                in descending order of priority,
                until, within a group, constraint violations
                are detected.

                *Deprecated since Pyomo 6.9.3*: The argument
                `separation_priority_order` is deprecated.
                Specify separation priorities by declaring, on your
                model, `Suffix` components with local name
                'pyros_separation_priority'.
                """
            ),
        ),
    )
    CONFIG.declare(
        "progress_logger",
        ConfigValue(
            default=default_pyros_solver_logger,
            domain=logger_domain,
            doc=(
                """
                Logger (or name thereof) used for reporting PyROS solver
                progress. If `None` or a `str` is provided, then
                ``progress_logger``
                is cast to ``logging.getLogger(progress_logger)``.
                In the default case, `progress_logger` is set to
                a :class:`pyomo.contrib.pyros.util.PreformattedLogger`
                object of level ``logging.INFO``.
                """
            ),
        ),
    )
    CONFIG.declare(
        "backup_local_solvers",
        ConfigValue(
            default=[],
            domain=SolverIterable(
                solver_desc="backup local solver",
                require_available=False,
                filter_by_availability=True,
            ),
            doc=(
                """
                Additional subordinate local NLP optimizers to invoke
                in the event the primary local NLP optimizer fails
                to solve a subproblem to an acceptable termination condition.
                """
            ),
        ),
    )
    CONFIG.declare(
        "backup_global_solvers",
        ConfigValue(
            default=[],
            domain=SolverIterable(
                solver_desc="backup global solver",
                require_available=False,
                filter_by_availability=True,
            ),
            doc=(
                """
                Additional subordinate global NLP optimizers to invoke
                in the event the primary global NLP optimizer fails
                to solve a subproblem to an acceptable termination condition.
                """
            ),
        ),
    )
    CONFIG.declare(
        "subproblem_file_directory",
        ConfigValue(
            default=None,
            domain=Path(),
            description=(
                """
                Directory to which to export subproblems not successfully
                solved to an acceptable termination condition.
                In the event ``keepfiles=True`` is specified, a str or
                path-like referring to an existing directory must be
                provided.
                """
            ),
        ),
    )
    CONFIG.declare(
        "subproblem_format_options",
        ConfigValue(
            default={"bar": {"symbolic_solver_labels": True}},
            # note: we leave all validation of the dict entries
            #       to ``BlockData.write()``
            domain=dict,
            description=(
                """
                File format options for writing/exporting subproblems
                that were not solved to an acceptable level
                if ``keepfiles=True`` is specified.
                Each entry of the dict should map a Pyomo WriterFactory
                format (e.g., 'bar' for BARON, 'gams' for GAMS)
                to a value for the argument ``io_options``
                to the method ``BlockData.write()``.
                """
            ),
        ),
    )

    # ================================================
    # === Advanced Options
    # ================================================
    CONFIG.declare(
        "bypass_local_separation",
        ConfigValue(
            default=False,
            domain=bool,
            description=(
                """
                This is an advanced option.
                Solve all separation subproblems with the subordinate global
                solver(s) only.
                This option is useful for expediting PyROS
                in the event that the subordinate global optimizer(s) provided
                can quickly solve separation subproblems to global optimality.
                """
            ),
        ),
    )
    CONFIG.declare(
        "bypass_global_separation",
        ConfigValue(
            default=False,
            domain=bool,
            doc=(
                """
                This is an advanced option.
                Solve all separation subproblems with the subordinate local
                solver(s) only.
                If `True` is chosen, then robustness of the final solution(s)
                returned by PyROS is not guaranteed, and a warning will
                be issued at termination.
                This option is useful for expediting PyROS
                in the event that the subordinate global optimizer provided
                cannot tractably solve separation subproblems to global
                optimality.
                """
            ),
        ),
    )
    CONFIG.declare(
        "p_robustness",
        ConfigValue(
            default={},
            domain=dict,
            doc=(
                """
                This is an advanced option.
                Add p-robustness constraints to all master subproblems.
                If an empty dict is provided, then p-robustness constraints
                are not added.
                Otherwise, the dict must map a `str` of value ``'rho'``
                to a non-negative `float`. PyROS automatically
                specifies ``1 + p_robustness['rho']``
                as an upper bound for the ratio of the
                objective function value under any PyROS-sampled uncertain
                parameter realization to the objective function under
                the nominal parameter realization.
                """
            ),
            visibility=1,
        ),
    )

    return CONFIG
