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

# pyros.py: Generalized Robust Cutting-Set Algorithm for Pyomo
import logging
from textwrap import indent, dedent, wrap
from pyomo.common.collections import Bunch, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue, In, NonNegativeFloat
from pyomo.core.base.block import Block
from pyomo.core.expr import value
from pyomo.core.base.var import Var, _VarData
from pyomo.core.base.param import Param, _ParamData
from pyomo.core.base.objective import Objective, maximize
from pyomo.contrib.pyros.util import a_logger, time_code, get_main_elapsed_time
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import (
    model_is_valid,
    recast_to_min_obj,
    add_decision_rule_constraints,
    add_decision_rule_variables,
    load_final_solution,
    pyrosTerminationCondition,
    ValidEnum,
    ObjectiveType,
    validate_uncertainty_set,
    identify_objective_functions,
    validate_kwarg_inputs,
    transform_to_standard_form,
    turn_bounds_to_constraints,
    replace_uncertain_bounds_with_constraints,
    IterationLogRecord,
    setup_pyros_logger,
    TimingData,
)
from pyomo.contrib.pyros.solve_data import ROSolveResults
from pyomo.contrib.pyros.pyros_algorithm_methods import ROSolver_iterative_solve
from pyomo.contrib.pyros.uncertainty_sets import uncertainty_sets
from pyomo.core.base import Constraint

from datetime import datetime


__version__ = "1.2.8"


default_pyros_solver_logger = setup_pyros_logger()


def _get_pyomo_version_info():
    """
    Get Pyomo version information.
    """
    import os
    import subprocess
    from pyomo.version import version

    pyomo_version = version
    commit_hash = "unknown"

    pyros_dir = os.path.join(*os.path.split(__file__)[:-1])
    commit_hash_command_args = [
        "git",
        "-C",
        f"{pyros_dir}",
        "rev-parse",
        "--short",
        "HEAD",
    ]
    try:
        commit_hash = (
            subprocess.check_output(commit_hash_command_args).decode("ascii").strip()
        )
    except subprocess.CalledProcessError:
        commit_hash = "unknown"

    return {"Pyomo version": pyomo_version, "Commit hash": commit_hash}


def NonNegIntOrMinusOne(obj):
    '''
    if obj is a non-negative int, return the non-negative int
    if obj is -1, return -1
    else, error
    '''
    ans = int(obj)
    if ans != float(obj) or (ans < 0 and ans != -1):
        raise ValueError("Expected non-negative int, but received %s" % (obj,))
    return ans


def PositiveIntOrMinusOne(obj):
    '''
    if obj is a positive int, return the int
    if obj is -1, return -1
    else, error
    '''
    ans = int(obj)
    if ans != float(obj) or (ans <= 0 and ans != -1):
        raise ValueError("Expected positive int, but received %s" % (obj,))
    return ans


class SolverResolvable(object):
    def __call__(self, obj):
        '''
        if obj is a string, return the Solver object for that solver name
        if obj is a Solver object, return a copy of the Solver
        if obj is a list, and each element of list is solver resolvable, return list of solvers
        '''
        if isinstance(obj, str):
            return SolverFactory(obj.lower())
        elif callable(getattr(obj, "solve", None)):
            return obj
        elif isinstance(obj, list):
            return [self(o) for o in obj]
        else:
            raise ValueError(
                "Expected a Pyomo solver or string object, "
                "instead received {1}".format(obj.__class__.__name__)
            )


class InputDataStandardizer(object):
    def __init__(self, ctype, cdatatype):
        self.ctype = ctype
        self.cdatatype = cdatatype

    def __call__(self, obj):
        if isinstance(obj, self.ctype):
            return list(obj.values())
        if isinstance(obj, self.cdatatype):
            return [obj]
        ans = []
        for item in obj:
            ans.extend(self.__call__(item))
        for _ in ans:
            assert isinstance(_, self.cdatatype)
        return ans


class PyROSConfigValue(ConfigValue):
    """
    Subclass of ``common.collections.ConfigValue``,
    with a few attributes added to facilitate documentation
    of the PyROS solver.
    An instance of this class is used for storing and
    documenting an argument to the PyROS solver.

    Attributes
    ----------
    is_optional : bool
        Argument is optional.
    document_default : bool, optional
        Document the default value of the argument
        in any docstring generated from this instance,
        or a `ConfigDict` object containing this instance.
    dtype_spec_str : None or str, optional
        String documenting valid types for this argument.
        If `None` is provided, then this string is automatically
        determined based on the `domain` argument to the
        constructor.

    NOTES
    -----
    Cleaner way to access protected attributes
    (particularly _doc, _description) inherited from ConfigValue?

    """

    def __init__(
        self,
        default=None,
        domain=None,
        description=None,
        doc=None,
        visibility=0,
        is_optional=True,
        document_default=True,
        dtype_spec_str=None,
    ):
        """Initialize self (see class docstring)."""

        # initialize base class attributes
        super(self.__class__, self).__init__(
            default=default,
            domain=domain,
            description=description,
            doc=doc,
            visibility=visibility,
        )

        self.is_optional = is_optional
        self.document_default = document_default

        if dtype_spec_str is None:
            self.dtype_spec_str = self.domain_name()
            # except AttributeError:
            #     self.dtype_spec_str = repr(self._domain)
        else:
            self.dtype_spec_str = dtype_spec_str


def pyros_config():
    CONFIG = ConfigDict('PyROS')

    # ================================================
    # === Options common to all solvers
    # ================================================
    CONFIG.declare(
        'time_limit',
        PyROSConfigValue(
            default=None,
            domain=NonNegativeFloat,
            doc=(
                """
                Wall time limit for the execution of the PyROS solver
                in seconds (including time spent by subsolvers).
                If `None` is provided, then no time limit is enforced.
                """
            ),
            is_optional=True,
            document_default=False,
            dtype_spec_str="None or NonNegativeFloat",
        ),
    )
    CONFIG.declare(
        'keepfiles',
        PyROSConfigValue(
            default=False,
            domain=bool,
            description=(
                """
                Export subproblems with a non-acceptable termination status
                for debugging purposes.
                If True is provided, then the argument `subproblem_file_directory`
                must also be specified.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        'tee',
        PyROSConfigValue(
            default=False,
            domain=bool,
            description="Output subordinate solver logs for all subproblems.",
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        'load_solution',
        PyROSConfigValue(
            default=True,
            domain=bool,
            description=(
                """
                Load final solution(s) found by PyROS to the deterministic model
                provided.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )

    # ================================================
    # === Required User Inputs
    # ================================================
    CONFIG.declare(
        "first_stage_variables",
        PyROSConfigValue(
            default=[],
            domain=InputDataStandardizer(Var, _VarData),
            description="First-stage (or design) variables.",
            is_optional=False,
            dtype_spec_str="list of Var",
        ),
    )
    CONFIG.declare(
        "second_stage_variables",
        PyROSConfigValue(
            default=[],
            domain=InputDataStandardizer(Var, _VarData),
            description="Second-stage (or control) variables.",
            is_optional=False,
            dtype_spec_str="list of Var",
        ),
    )
    CONFIG.declare(
        "uncertain_params",
        PyROSConfigValue(
            default=[],
            domain=InputDataStandardizer(Param, _ParamData),
            description=(
                """
                Uncertain model parameters.
                The `mutable` attribute for all uncertain parameter
                objects should be set to True.
                """
            ),
            is_optional=False,
            dtype_spec_str="list of Param",
        ),
    )
    CONFIG.declare(
        "uncertainty_set",
        PyROSConfigValue(
            default=None,
            domain=uncertainty_sets,
            description=(
                """
                Uncertainty set against which the
                final solution(s) returned by PyROS should be certified
                to be robust.
                """
            ),
            is_optional=False,
            dtype_spec_str="UncertaintySet",
        ),
    )
    CONFIG.declare(
        "local_solver",
        PyROSConfigValue(
            default=None,
            domain=SolverResolvable(),
            description="Subordinate local NLP solver.",
            is_optional=False,
            dtype_spec_str="Solver",
        ),
    )
    CONFIG.declare(
        "global_solver",
        PyROSConfigValue(
            default=None,
            domain=SolverResolvable(),
            description="Subordinate global NLP solver.",
            is_optional=False,
            dtype_spec_str="Solver",
        ),
    )
    # ================================================
    # === Optional User Inputs
    # ================================================
    CONFIG.declare(
        "objective_focus",
        PyROSConfigValue(
            default=ObjectiveType.nominal,
            domain=ValidEnum(ObjectiveType),
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
            is_optional=True,
            document_default=False,
            dtype_spec_str="ObjectiveType",
        ),
    )
    CONFIG.declare(
        "nominal_uncertain_param_vals",
        PyROSConfigValue(
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
            is_optional=True,
            document_default=True,
            dtype_spec_str="list of float",
        ),
    )
    CONFIG.declare(
        "decision_rule_order",
        PyROSConfigValue(
            default=0,
            domain=In([0, 1, 2]),
            description=(
                """
                Order (or degree) of the polynomial decision rule functions used
                for approximating the adjustability of the second stage
                variables with respect to the uncertain parameters.
                """
            ),
            doc=(
                """
                Order (or degree) of the polynomial decision rule functions used
                for approximating the adjustability of the second stage
                variables with respect to the uncertain parameters.
    
                Choices are:
    
                - 0: static recourse
                - 1: affine recourse
                - 2: quadratic recourse
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        "solve_master_globally",
        PyROSConfigValue(
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
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        "max_iter",
        PyROSConfigValue(
            default=-1,
            domain=PositiveIntOrMinusOne,
            description=(
                """
                Iteration limit. If -1 is provided, then no iteration
                limit is enforced.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="int",
        ),
    )
    CONFIG.declare(
        "robust_feasibility_tolerance",
        PyROSConfigValue(
            default=1e-4,
            domain=NonNegativeFloat,
            description=(
                """
                Relative tolerance for assessing maximal inequality
                constraint violations during the GRCS separation step.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        "separation_priority_order",
        PyROSConfigValue(
            default={},
            domain=dict,
            doc=(
                """
                Mapping from model inequality constraint names
                to positive integers specifying the priorities
                of their corresponding separation subproblems.
                A higher integer value indicates a higher priority.
                Constraints not referenced in the `dict` assume
                a priority of 0.
                Separation subproblems are solved in order of decreasing
                priority.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        "progress_logger",
        PyROSConfigValue(
            default=default_pyros_solver_logger,
            domain=a_logger,
            doc=(
                """
                Logger (or name thereof) used for reporting PyROS solver
                progress. If a `str` is specified, then ``progress_logger``
                is cast to ``logging.getLogger(progress_logger)``.
                In the default case, `progress_logger` is set to
                a :class:`pyomo.contrib.pyros.util.PreformattedLogger`
                object of level ``logging.INFO``.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="str or logging.Logger",
        ),
    )
    CONFIG.declare(
        "backup_local_solvers",
        PyROSConfigValue(
            default=[],
            domain=SolverResolvable(),
            doc=(
                """
                Additional subordinate local NLP optimizers to invoke
                in the event the primary local NLP optimizer fails
                to solve a subproblem to an acceptable termination condition.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="list of Solver",
        ),
    )
    CONFIG.declare(
        "backup_global_solvers",
        PyROSConfigValue(
            default=[],
            domain=SolverResolvable(),
            doc=(
                """
                Additional subordinate global NLP optimizers to invoke
                in the event the primary global NLP optimizer fails
                to solve a subproblem to an acceptable termination condition.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="list of Solver",
        ),
    )
    CONFIG.declare(
        "subproblem_file_directory",
        PyROSConfigValue(
            default=None,
            domain=str,
            description=(
                """
                Directory to which to export subproblems not successfully
                solved to an acceptable termination condition.
                In the event ``keepfiles=True`` is specified, a str or
                path-like referring to an existing directory must be
                provided.
                """
            ),
            is_optional=True,
            document_default=True,
            dtype_spec_str="None, str, or path-like",
        ),
    )

    # ================================================
    # === Advanced Options
    # ================================================
    CONFIG.declare(
        "bypass_local_separation",
        PyROSConfigValue(
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
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        "bypass_global_separation",
        PyROSConfigValue(
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
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )
    CONFIG.declare(
        "p_robustness",
        PyROSConfigValue(
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
            is_optional=True,
            document_default=True,
            dtype_spec_str=None,
        ),
    )

    return CONFIG


@SolverFactory.register(
    "pyros",
    doc="Robust optimization (RO) solver implementing "
    "the generalized robust cutting-set algorithm (GRCS)",
)
class PyROS(object):
    '''
    PyROS (Pyomo Robust Optimization Solver) implementing a
    generalized robust cutting-set algorithm (GRCS)
    to solve two-stage NLP optimization models under uncertainty.
    '''

    CONFIG = pyros_config()
    _LOG_LINE_LENGTH = 78

    def available(self, exception_flag=True):
        """Check if solver is available."""
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def license_is_valid(self):
        '''License for using PyROS'''
        return True

    # The Pyomo solver API expects that solvers support the context
    # manager API
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        pass

    def _log_intro(self, logger, **log_kwargs):
        """
        Log PyROS solver introductory messages.

        Parameters
        ----------
        logger : logging.Logger
            Logger through which to emit messages.
        **log_kwargs : dict, optional
            Keyword arguments to ``logger.log()`` callable.
            Should not include `msg`.
        """
        logger.log(msg="=" * self._LOG_LINE_LENGTH, **log_kwargs)
        logger.log(
            msg=f"PyROS: The Pyomo Robust Optimization Solver, v{self.version()}.",
            **log_kwargs,
        )

        # git_info_str = ", ".join(
        #     f"{field}: {val}" for field, val in _get_pyomo_git_info().items()
        # )
        version_info = _get_pyomo_version_info()
        version_info_str = ' ' * len("PyROS: ") + ("\n" + ' ' * len("PyROS: ")).join(
            f"{key}: {val}" for key, val in version_info.items()
        )
        logger.log(msg=version_info_str, **log_kwargs)
        logger.log(
            msg=(
                f"{' ' * len('PyROS:')} "
                f"Invoked at UTC {datetime.utcnow().isoformat()}"
            ),
            **log_kwargs,
        )
        logger.log(msg="", **log_kwargs)
        logger.log(
            msg=("Developed by: Natalie M. Isenberg (1), Jason A. F. Sherman (1),"),
            **log_kwargs,
        )
        logger.log(
            msg=(
                f"{' ' * len('Developed by:')} "
                "John D. Siirola (2), Chrysanthos E. Gounaris (1)"
            ),
            **log_kwargs,
        )
        logger.log(
            msg=(
                "(1) Carnegie Mellon University, " "Department of Chemical Engineering"
            ),
            **log_kwargs,
        )
        logger.log(
            msg="(2) Sandia National Laboratories, Center for Computing Research",
            **log_kwargs,
        )
        logger.log(msg="", **log_kwargs)
        logger.log(
            msg=(
                "The developers gratefully acknowledge support "
                "from the U.S. Department"
            ),
            **log_kwargs,
        )
        logger.log(
            msg=(
                "of Energy's "
                "Institute for the Design of Advanced Energy Systems (IDAES)."
            ),
            **log_kwargs,
        )
        logger.log(msg="=" * self._LOG_LINE_LENGTH, **log_kwargs)

    def _log_disclaimer(self, logger, **log_kwargs):
        """
        Log PyROS solver disclaimer messages.

        Parameters
        ----------
        logger : logging.Logger
            Logger through which to emit messages.
        **log_kwargs : dict, optional
            Keyword arguments to ``logger.log()`` callable.
            Should not include `msg`.
        """
        disclaimer_header = " DISCLAIMER ".center(self._LOG_LINE_LENGTH, "=")

        logger.log(msg=disclaimer_header, **log_kwargs)
        logger.log(msg="PyROS is still under development. ", **log_kwargs)
        logger.log(
            msg=(
                "Please provide feedback and/or report any issues by creating "
                "a ticket at"
            ),
            **log_kwargs,
        )
        logger.log(msg="https://github.com/Pyomo/pyomo/issues/new/choose", **log_kwargs)
        logger.log(msg="=" * self._LOG_LINE_LENGTH, **log_kwargs)

    def _log_config(self, logger, config, exclude_options=None, **log_kwargs):
        """
        Log PyROS solver options.

        Parameters
        ----------
        logger : logging.Logger
            Logger for the solver options.
        config : ConfigDict
            PyROS solver options.
        exclude_options : None or iterable of str, optional
            Options (keys of the ConfigDict) to exclude from
            logging. If `None` passed, then the names of the
            required arguments to ``self.solve()`` are skipped.
        **log_kwargs : dict, optional
            Keyword arguments to each statement of ``logger.log()``.
        """
        # log solver options
        if exclude_options is None:
            exclude_options = [
                "first_stage_variables",
                "second_stage_variables",
                "uncertain_params",
                "uncertainty_set",
                "local_solver",
                "global_solver",
            ]

        logger.log(msg="Solver options:", **log_kwargs)
        for key, val in config.items():
            if key not in exclude_options:
                logger.log(msg=f" {key}={val!r}", **log_kwargs)
        logger.log(msg="-" * self._LOG_LINE_LENGTH, **log_kwargs)

    def solve(
        self,
        model,
        first_stage_variables,
        second_stage_variables,
        uncertain_params,
        uncertainty_set,
        local_solver,
        global_solver,
        **kwds,
    ):
        """Solve a model.

        Parameters
        ----------
        model: ConcreteModel
            The deterministic model.
        first_stage_variables: list of Var
            First-stage model variables (or design variables).
        second_stage_variables: list of Var
            Second-stage model variables (or control variables).
        uncertain_params: list of Param
            Uncertain model parameters.
            The `mutable` attribute for every uncertain parameter
            objects must be set to True.
        uncertainty_set: UncertaintySet
            Uncertainty set against which the solution(s) returned
            will be confirmed to be robust.
        local_solver: Solver
            Subordinate local NLP solver.
        global_solver: Solver
            Subordinate global NLP solver.

        Returns
        -------
        return_soln : ROSolveResults
            Summary of PyROS termination outcome.

        """

        # === Add the explicit arguments to the config
        config = self.CONFIG(kwds.pop('options', {}))
        config.first_stage_variables = first_stage_variables
        config.second_stage_variables = second_stage_variables
        config.uncertain_params = uncertain_params
        config.uncertainty_set = uncertainty_set
        config.local_solver = local_solver
        config.global_solver = global_solver

        dev_options = kwds.pop('dev_options', {})
        config.set_value(kwds)
        config.set_value(dev_options)

        model = model

        # === Validate kwarg inputs
        validate_kwarg_inputs(model, config)

        # === Validate ability of grcs RO solver to handle this model
        if not model_is_valid(model):
            raise AttributeError(
                "This model structure is not currently handled by the ROSolver."
            )

        # === Define nominal point if not specified
        if len(config.nominal_uncertain_param_vals) == 0:
            config.nominal_uncertain_param_vals = list(
                p.value for p in config.uncertain_params
            )
        elif len(config.nominal_uncertain_param_vals) != len(config.uncertain_params):
            raise AttributeError(
                "The nominal_uncertain_param_vals list must be the same length"
                "as the uncertain_params list"
            )

        # === Create data containers
        model_data = ROSolveResults()
        model_data.timing = Bunch()

        # === Start timer, run the algorithm
        model_data.timing = TimingData()
        with time_code(
            timing_data_obj=model_data.timing,
            code_block_name="main",
            is_main_timer=True,
        ):
            # output intro and disclaimer
            self._log_intro(logger=config.progress_logger, level=logging.INFO)
            self._log_disclaimer(logger=config.progress_logger, level=logging.INFO)
            self._log_config(
                logger=config.progress_logger,
                config=config,
                exclude_options=None,
                level=logging.INFO,
            )

            # begin preprocessing
            config.progress_logger.info("Preprocessing...")
            model_data.timing.start_timer("main.preprocessing")

            # === A block to hold list-type data to make cloning easy
            util = Block(concrete=True)
            util.first_stage_variables = config.first_stage_variables
            util.second_stage_variables = config.second_stage_variables
            util.uncertain_params = config.uncertain_params

            model_data.util_block = unique_component_name(model, 'util')
            model.add_component(model_data.util_block, util)
            # Note:  model.component(model_data.util_block) is util

            # === Validate uncertainty set happens here, requires util block for Cardinality and FactorModel sets
            validate_uncertainty_set(config=config)

            # === Leads to a logger warning here for inactive obj when cloning
            model_data.original_model = model
            # === For keeping track of variables after cloning
            cname = unique_component_name(model_data.original_model, 'tmp_var_list')
            src_vars = list(model_data.original_model.component_data_objects(Var))
            setattr(model_data.original_model, cname, src_vars)
            model_data.working_model = model_data.original_model.clone()

            # identify active objective function
            # (there should only be one at this point)
            # recast to minimization if necessary
            active_objs = list(
                model_data.working_model.component_data_objects(
                    Objective, active=True, descend_into=True
                )
            )
            assert len(active_objs) == 1
            active_obj = active_objs[0]
            active_obj_original_sense = active_obj.sense
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

            # === Assuming all other Var objects in the model are state variables
            fsv = ComponentSet(model_data.working_model.util.first_stage_variables)
            ssv = ComponentSet(model_data.working_model.util.second_stage_variables)
            sv = ComponentSet()
            model_data.working_model.util.state_vars = []
            for v in model_data.working_model.component_data_objects(Var):
                if v not in fsv and v not in ssv and v not in sv:
                    model_data.working_model.util.state_vars.append(v)
                    sv.add(v)

            # Bounds on second stage variables and state variables are separation objectives,
            #  they are brought in this was as explicit constraints
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

            model_data.timing.stop_timer("main.preprocessing")
            preprocessing_time = model_data.timing.get_total_time("main.preprocessing")
            config.progress_logger.info(
                f"Done preprocessing; required wall time of "
                f"{preprocessing_time:.3f}s."
            )

            # === Solve and load solution into model
            pyros_soln, final_iter_separation_solns = ROSolver_iterative_solve(
                model_data, config
            )
            IterationLogRecord.log_header_rule(config.progress_logger.info)

            return_soln = ROSolveResults()
            if pyros_soln is not None and final_iter_separation_solns is not None:
                if config.load_solution and (
                    pyros_soln.pyros_termination_condition
                    is pyrosTerminationCondition.robust_optimal
                    or pyros_soln.pyros_termination_condition
                    is pyrosTerminationCondition.robust_feasible
                ):
                    load_final_solution(model_data, pyros_soln.master_soln, config)

                # account for sense of the original model objective
                # when reporting the final PyROS (master) objective,
                # since maximization objective is changed to
                # minimization objective during preprocessing
                if config.objective_focus == ObjectiveType.nominal:
                    return_soln.final_objective_value = (
                        active_obj_original_sense
                        * value(pyros_soln.master_soln.master_model.obj)
                    )
                elif config.objective_focus == ObjectiveType.worst_case:
                    return_soln.final_objective_value = (
                        active_obj_original_sense
                        * value(pyros_soln.master_soln.master_model.zeta)
                    )
                return_soln.pyros_termination_condition = (
                    pyros_soln.pyros_termination_condition
                )
                return_soln.iterations = pyros_soln.total_iters + 1

                # === Remove util block
                model.del_component(model_data.util_block)

                del pyros_soln.util_block
                del pyros_soln.working_model
            else:
                return_soln.final_objective_value = None
                return_soln.pyros_termination_condition = (
                    pyrosTerminationCondition.robust_infeasible
                )
                return_soln.iterations = 0

        return_soln.config = config
        return_soln.time = model_data.timing.get_total_time("main")

        # log termination-related messages
        config.progress_logger.info(return_soln.pyros_termination_condition.message)
        config.progress_logger.info("-" * self._LOG_LINE_LENGTH)
        config.progress_logger.info(f"Timing breakdown:\n\n{model_data.timing}")
        config.progress_logger.info("-" * self._LOG_LINE_LENGTH)
        config.progress_logger.info(return_soln)
        config.progress_logger.info("-" * self._LOG_LINE_LENGTH)
        config.progress_logger.info("All done. Exiting PyROS.")
        config.progress_logger.info("=" * self._LOG_LINE_LENGTH)

        return return_soln


def _generate_filtered_docstring():
    """
    Add Numpy-style 'Keyword arguments' section to `PyROS.solve()`
    docstring.
    """
    cfg = PyROS.CONFIG()

    # mandatory args already documented
    exclude_args = [
        "first_stage_variables",
        "second_stage_variables",
        "uncertain_params",
        "uncertainty_set",
        "local_solver",
        "global_solver",
    ]

    indent_by = 8
    width = 72
    before = PyROS.solve.__doc__
    section_name = "Keyword Arguments"

    indent_str = ' ' * indent_by
    wrap_width = width - indent_by
    cfg = pyros_config()

    arg_docs = []

    def wrap_doc(doc, indent_by, width):
        """
        Wrap a string, accounting for paragraph
        breaks ('\n\n') and bullet points (paragraphs
        which, when dedented, are such that each line
        starts with '- ' or '  ').
        """
        paragraphs = doc.split("\n\n")
        wrapped_pars = []
        for par in paragraphs:
            lines = dedent(par).split("\n")
            has_bullets = all(
                line.startswith("- ") or line.startswith("  ")
                for line in lines
                if line != ""
            )
            if has_bullets:
                # obtain strings of each bullet point
                # (dedented, bullet dash and bullet indent removed)
                bullet_groups = []
                new_group = False
                group = ""
                for line in lines:
                    new_group = line.startswith("- ")
                    if new_group:
                        bullet_groups.append(group)
                        group = ""
                    new_line = line[2:]
                    group += f"{new_line}\n"
                if group != "":
                    # ensure last bullet not skipped
                    bullet_groups.append(group)

                # first entry is just ''; remove
                bullet_groups = bullet_groups[1:]

                # wrap each bullet point, then add bullet
                # and indents as necessary
                wrapped_groups = []
                for group in bullet_groups:
                    wrapped_groups.append(
                        "\n".join(
                            f"{'- ' if idx == 0 else '  '}{line}"
                            for idx, line in enumerate(
                                wrap(group, width - 2 - indent_by)
                            )
                        )
                    )

                # now combine bullets into single 'paragraph'
                wrapped_pars.append(
                    indent("\n".join(wrapped_groups), prefix=' ' * indent_by)
                )
            else:
                wrapped_pars.append(
                    indent(
                        "\n".join(wrap(dedent(par), width=width - indent_by)),
                        prefix=' ' * indent_by,
                    )
                )

        return "\n\n".join(wrapped_pars)

    section_header = indent(f"{section_name}\n" + "-" * len(section_name), indent_str)
    for key, itm in cfg._data.items():
        if key in exclude_args:
            continue
        arg_name = key
        arg_dtype = itm.dtype_spec_str

        if itm.is_optional:
            if itm.document_default:
                optional_str = f", default={repr(itm._default)}"
            else:
                optional_str = ", optional"
        else:
            optional_str = ""

        arg_header = f"{indent_str}{arg_name} : {arg_dtype}{optional_str}"

        # dedented_doc_str = dedent(itm.doc).replace("\n", ' ').strip()
        if itm._doc is not None:
            raw_arg_desc = itm._doc
        else:
            raw_arg_desc = itm._description

        arg_description = wrap_doc(
            raw_arg_desc, width=wrap_width, indent_by=indent_by + 4
        )

        arg_docs.append(f"{arg_header}\n{arg_description}")

    kwargs_section_doc = "\n".join([section_header] + arg_docs)

    return f"{before}\n{kwargs_section_doc}\n"


PyROS.solve.__doc__ = _generate_filtered_docstring()
