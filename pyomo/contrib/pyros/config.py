"""
Interfaces for managing PyROS solver options.
"""


from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    In,
    NonNegativeFloat,
)
from pyomo.core.base import (
    Var,
    _VarData,
)
from pyomo.core.base.param import (
    Param,
    _ParamData,
)
from pyomo.opt import SolverFactory
from pyomo.contrib.pyros.util import (
    a_logger,
    ObjectiveType,
    setup_pyros_logger,
    ValidEnum,
)
from pyomo.contrib.pyros.uncertainty_sets import uncertainty_sets


default_pyros_solver_logger = setup_pyros_logger()


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
        if obj is a list, and each element of list is solver resolvable,
        return list of solvers
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
                "instead received {0}".format(obj.__class__.__name__)
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
                If True is provided, then the argument `subproblem_file_directory`
                must also be specified.
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
                Load final solution(s) found by PyROS to the deterministic model
                provided.
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
            domain=InputDataStandardizer(Var, _VarData),
            description="First-stage (or design) variables.",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "second_stage_variables",
        ConfigValue(
            default=[],
            domain=InputDataStandardizer(Var, _VarData),
            description="Second-stage (or control) variables.",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "uncertain_params",
        ConfigValue(
            default=[],
            domain=InputDataStandardizer(Param, _ParamData),
            description=(
                """
                Uncertain model parameters.
                The `mutable` attribute for all uncertain parameter
                objects should be set to True.
                """
            ),
            visibility=1,
        ),
    )
    CONFIG.declare(
        "uncertainty_set",
        ConfigValue(
            default=None,
            domain=uncertainty_sets,
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
            domain=SolverResolvable(),
            description="Subordinate local NLP solver.",
            visibility=1,
        ),
    )
    CONFIG.declare(
        "global_solver",
        ConfigValue(
            default=None,
            domain=SolverResolvable(),
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
            domain=PositiveIntOrMinusOne,
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
        ),
    )
    CONFIG.declare(
        "progress_logger",
        ConfigValue(
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
        ),
    )
    CONFIG.declare(
        "backup_local_solvers",
        ConfigValue(
            default=[],
            domain=SolverResolvable(),
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
            domain=SolverResolvable(),
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
        ),
    )

    return CONFIG
