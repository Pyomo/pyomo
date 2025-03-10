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

from pyomo.common.config import (
    ConfigBlock,
    ConfigList,
    ConfigValue,
    In,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import a_logger, _DoNothing
from pyomo.util.config_domains import ComponentDataSet
from pyomo.core.base import LogicalConstraint
from pyomo.gdp.disjunct import Disjunction

_supported_algorithms = {
    'LOA': ('gdpopt.loa', 'Logic-based Outer Approximation'),
    'GLOA': ('gdpopt.gloa', 'Global Logic-based Outer Approximation'),
    'LBB': ('gdpopt.lbb', 'Logic-based Branch and Bound'),
    'RIC': ('gdpopt.ric', 'Relaxation with Integer Cuts'),
    'enumerate': ('gdpopt.enumerate', 'Enumeration of discrete solutions'),
}


def _strategy_deprecation(strategy):
    deprecation_warning(
        "The argument 'strategy' has been deprecated in favor of 'algorithm.'",
        version="6.4.2",
    )
    return In(_supported_algorithms)(strategy)


def _init_strategy_deprecation(strategy):
    deprecation_warning(
        "The argument 'init_strategy' has been deprecated "
        "in favor of 'init_algorithm.'",
        version="6.4.2",
    )
    return In(valid_init_strategies)(strategy)


def _get_algorithm_config():
    CONFIG = ConfigBlock("GDPoptAlgorithm")
    CONFIG.declare(
        "strategy",
        ConfigValue(
            default=None,
            domain=_strategy_deprecation,
            description="DEPRECATED: Please use 'algorithm' instead.",
        ),
    )
    CONFIG.declare(
        "algorithm",
        ConfigValue(
            default=None,
            domain=In(_supported_algorithms),
            description="Algorithm to use.",
        ),
    )
    return CONFIG


def _add_common_configs(CONFIG):
    CONFIG.declare(
        "iterlim",
        ConfigValue(
            default=None, domain=NonNegativeInt, description="Iteration limit."
        ),
    )
    CONFIG.declare(
        "time_limit",
        ConfigValue(
            default=None,
            domain=PositiveInt,
            description="Time limit (seconds, default=600)",
            doc="""
            Seconds allowed until terminated. Note that the time limit can
            currently only be enforced between subsolver invocations. You may
            need to set subsolver time limits as well.""",
        ),
    )
    CONFIG.declare(
        "tee",
        ConfigValue(
            default=False, description="Stream output to terminal.", domain=bool
        ),
    )
    CONFIG.declare(
        "logger",
        ConfigValue(
            default='pyomo.contrib.gdpopt',
            description="The logger object or name to use for reporting.",
            domain=a_logger,
        ),
    )


def _add_nlp_solve_configs(CONFIG, default_nlp_init_method):
    # All of these config options are expected if the algorithm solves NLP
    # subproblems.
    CONFIG.declare(
        "integer_tolerance",
        ConfigValue(default=1e-5, description="Tolerance on integral values."),
    )
    CONFIG.declare(
        "constraint_tolerance",
        ConfigValue(
            default=1e-6,
            description="""
            Tolerance on constraint satisfaction.
    
            Increasing this tolerance corresponds to being more conservative in
            declaring the model or an NLP subproblem to be infeasible.
            """,
        ),
    )
    CONFIG.declare(
        "variable_tolerance",
        ConfigValue(default=1e-8, description="Tolerance on variable bounds."),
    )
    CONFIG.declare(
        "subproblem_initialization_method",
        ConfigValue(
            default=default_nlp_init_method,
            description=""""
            callback to specify custom routines to initialize the
            (MI)NLP subproblems.""",
            doc="""
            Callback to specify custom routines for initializing the (MI)NLP
            subproblems. This method is called after the discrete problem solution
            is fixed in the subproblem and before the subproblem is solved (or
            pre-solved).
    
            For algorithms with a discrete problem relaxation:
            This method accepts three arguments: the solver object, the subproblem
            GDPopt utility block and the discrete problem GDPopt utility block. The
            discrete problem contains the most recent discrete problem solution.
    
            For algorithms without a discrete problem relaxation:
            This method accepts four arguments: the list of Disjuncts that are
            currently fixed as being active, a list of values for the non-indicator
            BooleanVars (empty if force_nlp_subproblem=False), and a list of
            values for the integer vars (also empty if force_nlp_subproblem=False),
            and last the subproblem GDPopt utility block.
    
            The return of this method will be unused: The method should directly
            set the value of the variables on the subproblem
            """,
        ),
    )
    CONFIG.declare(
        "call_before_subproblem_solve",
        ConfigValue(
            default=_DoNothing,
            description="callback hook before calling the subproblem solver",
            doc="""
            Callback called right before the (MI)NLP subproblem is solved.
            Takes three arguments: The solver object, the subproblem and the
            GDPopt utility block on the subproblem.
    
            Note that unless you are *very* confident in what you are doing, the
            subproblem should not be modified in this callback: it should be used
            to interrogate the problem only.
    
            To initialize the problem before it is solved, please specify a method
            in the 'subproblem_initialization_method' argument.
            """,
        ),
    )
    CONFIG.declare(
        "call_after_subproblem_solve",
        ConfigValue(
            default=_DoNothing,
            description="""
            callback hook after a solution of the
            "nonlinear subproblem""",
            doc="""
            Callback called right after the (MI)NLP subproblem is solved.
            Takes three arguments: The solver object, the subproblem, and the
            GDPopt utility block on the subproblem.
    
            Note that unless you are *very* confident in what you are doing, the
            subproblem should not be modified in this callback: it should be used
            to interrogate the problem only.
            """,
        ),
    )
    CONFIG.declare(
        "call_after_subproblem_feasible",
        ConfigValue(
            default=_DoNothing,
            description="""
            callback hook after feasible solution of
            the nonlinear subproblem""",
            doc="""
            Callback called right after the (MI)NLP subproblem is solved,
            if it was feasible. Takes three arguments: The solver object, the
            subproblem and the GDPopt utility block on the subproblem.
    
            Note that unless you are *very* confident in what you are doing, the
            subproblem should not be modified in this callback: it should be used
            to interrogate the problem only.
            """,
        ),
    )
    CONFIG.declare(
        "force_subproblem_nlp",
        ConfigValue(
            default=False,
            description="""Force subproblems to be NLP, even if discrete variables
            exist.""",
        ),
    )
    CONFIG.declare(
        "subproblem_presolve",
        ConfigValue(
            default=True,
            description="""
        Flag to enable or disable subproblem presolve.
        Default=True.""",
            domain=bool,
        ),
    )
    CONFIG.declare(
        "tighten_nlp_var_bounds",
        ConfigValue(
            default=False,
            description="""
            Whether or not to do feasibility-based bounds tightening
            on the variables in the NLP subproblem before solving it.""",
            domain=bool,
        ),
    )
    CONFIG.declare(
        "round_discrete_vars",
        ConfigValue(
            default=True,
            description="""Flag to round subproblem discrete variable values to the
            nearest integer. Rounding is done before fixing disjuncts.""",
        ),
    )
    CONFIG.declare(
        "max_fbbt_iterations",
        ConfigValue(
            default=3,
            description="""
            Maximum number of feasibility-based bounds tightening
            iterations to do during NLP subproblem preprocessing.""",
            domain=PositiveInt,
        ),
    )


def _add_oa_configs(CONFIG):
    _add_nlp_solve_configs(
        CONFIG, default_nlp_init_method=restore_vars_to_original_values
    )

    CONFIG.declare(
        "init_strategy",
        ConfigValue(
            default=None,
            domain=_init_strategy_deprecation,
            description="DEPRECATED: Please use 'init_algorithm' instead.",
        ),
    )
    CONFIG.declare(
        "init_algorithm",
        ConfigValue(
            default="set_covering",
            domain=In(valid_init_strategies),
            description="Initialization algorithm to use.",
            doc="""
            Selects the initialization algorithm to use when generating
            the initial cuts to construct the discrete problem.""",
        ),
    )
    CONFIG.declare(
        "custom_init_disjuncts",
        ConfigList(
            # domain=ComponentSets of Disjuncts,
            default=None,
            description="List of disjunct sets to use for initialization.",
        ),
    )
    CONFIG.declare(
        "max_slack",
        ConfigValue(
            default=1000,
            domain=NonNegativeFloat,
            description="Upper bound on slack variables for OA",
        ),
    )
    CONFIG.declare(
        "OA_penalty_factor",
        ConfigValue(
            default=1000,
            domain=NonNegativeFloat,
            description="""
            Penalty multiplication term for slack variables on the
            objective value.""",
        ),
    )
    CONFIG.declare(
        "set_cover_iterlim",
        ConfigValue(
            default=8,
            domain=NonNegativeInt,
            description="Limit on the number of set covering iterations.",
        ),
    )
    CONFIG.declare(
        "discrete_problem_transformation",
        ConfigValue(
            default='gdp.bigm',
            description="""
            Name of the transformation to use to transform the
            discrete problem from a GDP to an algebraic model.""",
        ),
    )
    CONFIG.declare(
        "call_before_discrete_problem_solve",
        ConfigValue(
            default=_DoNothing,
            description="callback hook before calling the discrete problem solver",
            doc="""
            Callback called right before the MILP discrete problem is solved.
            Takes three arguments: The solver object, the discrete problem, and the
            GDPopt utility block on the discrete problem.
    
            Note that unless you are *very* confident in what you are doing, the
            problem should not be modified in this callback: it should be used
            to interrogate the problem only.
            """,
        ),
    )
    CONFIG.declare(
        "call_after_discrete_problem_solve",
        ConfigValue(
            default=_DoNothing,
            description="callback hook after a solution of the discrete problem",
            doc="""
            Callback called right after the MILP discrete problem is solved.
            Takes three arguments: The solver object, the discrete problem, and the
            GDPopt utility block on the discrete problem.
    
            Note that unless you are *very* confident in what you are doing, the
            problem should not be modified in this callback: it should be used
            to interrogate the problem only.
            """,
        ),
    )
    CONFIG.declare(
        "call_before_master_solve",
        ConfigValue(
            default=_DoNothing,
            description="DEPRECATED: Please use "
            "'call_before_discrete_problem_solve'",
        ),
    )
    CONFIG.declare(
        "call_after_master_solve",
        ConfigValue(
            default=_DoNothing,
            description="DEPRECATED: Please use 'call_after_discrete_problem_solve'",
        ),
    )
    CONFIG.declare(
        "mip_presolve",
        ConfigValue(
            default=True,
            description="""
        Flag to enable or disable GDPopt MIP presolve.
        Default=True.""",
            domain=bool,
        ),
    )
    CONFIG.declare(
        "calc_disjunctive_bounds",
        ConfigValue(
            default=False,
            description="""
        Calculate special disjunctive variable bounds for GLOA.
        False by default.""",
            domain=bool,
        ),
    )
    CONFIG.declare(
        "obbt_disjunctive_bounds",
        ConfigValue(
            default=False,
            description="""
            Use optimality-based bounds tightening rather than feasibility-based
            bounds tightening to compute disjunctive variable bounds. False by
            default.""",
            domain=bool,
        ),
    )


def _add_BB_configs(CONFIG):
    CONFIG.declare(
        "check_sat",
        ConfigValue(
            default=False,
            domain=bool,
            description="""
            When True, GDPopt-LBB will check satisfiability
            at each node via the pyomo.contrib.satsolver interface""",
        ),
    )
    CONFIG.declare(
        "solve_local_rnGDP",
        ConfigValue(
            default=False,
            domain=bool,
            description="""
            When True, GDPopt-LBB will solve a local MINLP at each node.""",
        ),
    )


def _add_mip_solver_configs(CONFIG):
    CONFIG.declare(
        "mip_solver",
        ConfigValue(
            default="gurobi",
            description="""
            Mixed-integer linear solver to use. Note that no persistent solvers
            other than the auto-persistent solvers in the APPSI package are
            supported.""",
        ),
    )
    CONFIG.declare(
        "mip_solver_args",
        ConfigBlock(
            description="""
            Keyword arguments to send to the MILP subsolver solve() invocation""",
            implicit=True,
        ),
    )


def _add_nlp_solver_configs(CONFIG, default_solver):
    CONFIG.declare(
        "nlp_solver",
        ConfigValue(
            default=default_solver,
            description="""
            Nonlinear solver to use. Note that no persistent solvers
            other than the auto-persistent solvers in the APPSI package are
            supported.""",
        ),
    )
    CONFIG.declare(
        "nlp_solver_args",
        ConfigBlock(
            description="""
            Keyword arguments to send to the NLP subsolver solve() invocation""",
            implicit=True,
        ),
    )
    CONFIG.declare(
        "minlp_solver",
        ConfigValue(
            default="baron",
            description="""
            Mixed-integer nonlinear solver to use. Note that no persistent solvers
            other than the auto-persistent solvers in the APPSI package are
            supported.""",
        ),
    )
    CONFIG.declare(
        "minlp_solver_args",
        ConfigBlock(
            description="""
            Keyword arguments to send to the MINLP subsolver solve() invocation""",
            implicit=True,
        ),
    )
    CONFIG.declare(
        "local_minlp_solver",
        ConfigValue(
            default="bonmin",
            description="""
            Mixed-integer nonlinear solver to use. Note that no persistent solvers
            other than the auto-persistent solvers in the APPSI package are
            supported.""",
        ),
    )
    CONFIG.declare(
        "local_minlp_solver_args",
        ConfigBlock(
            description="""
            Keyword arguments to send to the local MINLP subsolver solve()
            invocation""",
            implicit=True,
        ),
    )
    CONFIG.declare(
        "small_dual_tolerance",
        ConfigValue(
            default=1e-8,
            description="""
            When generating cuts, small duals multiplied by expressions can
            cause problems. Exclude all duals smaller in absolute value than the
            following.""",
        ),
    )


def _add_tolerance_configs(CONFIG):
    CONFIG.declare(
        "bound_tolerance",
        ConfigValue(
            default=1e-6,
            domain=NonNegativeFloat,
            description="Tolerance for bound convergence.",
        ),
    )


def _add_ldsda_configs(CONFIG):
    CONFIG.declare(
        "direction_norm",
        ConfigValue(
            default='L2',
            domain=In(['L2', 'Linf']),
            description="The norm to use for the search direction",
        ),
    )
    CONFIG.declare(
        "starting_point",
        ConfigValue(default=None, description="The value list of external variables."),
    )
    CONFIG.declare(
        "logical_constraint_list",
        ConfigValue(
            default=None,
            domain=ComponentDataSet(LogicalConstraint),
            description="""
            The list of logical constraints to be reformulated into external variables.
            The logical constraints should be in the same order of provided starting point.
            The provided logical constraints should be ExactlyExpressions.""",
        ),
    )
    CONFIG.declare(
        "disjunction_list",
        ConfigValue(
            default=None,
            domain=ComponentDataSet(Disjunction),
            description="""
            The list of disjunctions to be reformulated into external variables.
            The disjunctions should be in the same order of provided starting point.
            """,
        ),
    )
