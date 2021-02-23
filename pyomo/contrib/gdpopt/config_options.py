#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.config import (ConfigValue, NonNegativeInt, 
                                 In, PositiveInt, NonNegativeFloat,
                                 ConfigBlock, ConfigList)
from pyomo.contrib.gdpopt.master_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import _DoNothing, a_logger


def _get_GDPopt_config():
    _supported_strategies = {
        'LOA',  # Logic-based outer approximation
        'GLOA',  # Global logic-based outer approximation
        'LBB',  # Logic-based branch-and-bound
    }
    CONFIG = ConfigBlock("GDPopt")
    CONFIG.declare("iterlim", ConfigValue(
        default=100, domain=NonNegativeInt,
        description="Iteration limit."
    ))
    CONFIG.declare("time_limit", ConfigValue(
        default=600,
        domain=PositiveInt,
        description="Time limit (seconds, default=600)",
        doc="Seconds allowed until terminated. Note that the time limit can "
            "currently only be enforced between subsolver invocations. You may "
            "need to set subsolver time limits as well."
    ))
    CONFIG.declare("strategy", ConfigValue(
        default="LOA", domain=In(_supported_strategies),
        description="Decomposition strategy to use."
    ))
    CONFIG.declare("tee", ConfigValue(
        default=False,
        description="Stream output to terminal.",
        domain=bool
    ))
    CONFIG.declare("logger", ConfigValue(
        default='pyomo.contrib.gdpopt',
        description="The logger object or name to use for reporting.",
        domain=a_logger
    ))
    _add_OA_configs(CONFIG)
    _add_BB_configs(CONFIG)
    _add_subsolver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    return CONFIG


def _add_OA_configs(CONFIG):
    CONFIG.declare("init_strategy", ConfigValue(
        default="set_covering", domain=In(valid_init_strategies.keys()),
        description="Initialization strategy to use.",
        doc="Selects the initialization strategy to use when generating "
            "the initial cuts to construct the master problem."
    ))
    CONFIG.declare("custom_init_disjuncts", ConfigList(
        # domain=ComponentSets of Disjuncts,
        default=None,
        description="List of disjunct sets to use for initialization."
    ))
    CONFIG.declare("max_slack", ConfigValue(
        default=1000, domain=NonNegativeFloat,
        description="Upper bound on slack variables for OA"
    ))
    CONFIG.declare("OA_penalty_factor", ConfigValue(
        default=1000, domain=NonNegativeFloat,
        description="Penalty multiplication term for slack variables on the "
                    "objective value."
    ))
    CONFIG.declare("set_cover_iterlim", ConfigValue(
        default=8, domain=NonNegativeInt,
        description="Limit on the number of set covering iterations."
    ))
    CONFIG.declare("call_before_master_solve", ConfigValue(
        default=_DoNothing,
        description="callback hook before calling the master problem solver"
    ))
    CONFIG.declare("call_after_master_solve", ConfigValue(
        default=_DoNothing,
        description="callback hook after a solution of the master problem"
    ))
    CONFIG.declare("call_before_subproblem_solve", ConfigValue(
        default=_DoNothing,
        description="callback hook before calling the subproblem solver"
    ))
    CONFIG.declare("call_after_subproblem_solve", ConfigValue(
        default=_DoNothing,
        description="callback hook after a solution of the "
                    "nonlinear subproblem"
    ))
    CONFIG.declare("call_after_subproblem_feasible", ConfigValue(
        default=_DoNothing,
        description="callback hook after feasible solution of "
                    "the nonlinear subproblem"
    ))
    CONFIG.declare("algorithm_stall_after", ConfigValue(
        default=2,
        description="number of non-improving master iterations after which "
                    "the algorithm will stall and exit."
    ))
    CONFIG.declare("round_discrete_vars", ConfigValue(
        default=True,
        description="flag to round subproblem discrete variable values to the nearest integer. "
                    "Rounding is done before fixing disjuncts."
    ))
    CONFIG.declare("force_subproblem_nlp", ConfigValue(
        default=False,
        description="Force subproblems to be NLP, even if discrete variables exist."
    ))
    CONFIG.declare("mip_presolve", ConfigValue(
        default=True,
        description="Flag to enable or diable GDPopt MIP presolve. Default=True.",
        domain=bool
    ))
    CONFIG.declare("subproblem_presolve", ConfigValue(
        default=True,
        description="Flag to enable or disable subproblem presolve. Default=True.",
        domain=bool
    ))
    CONFIG.declare("calc_disjunctive_bounds", ConfigValue(
        default=False,
        description="Calculate special disjunctive variable bounds for GLOA. False by default.",
        domain=bool
    ))
    CONFIG.declare("obbt_disjunctive_bounds", ConfigValue(
        default=False,
        description="Use optimality-based bounds tightening rather than feasibility-based bounds tightening "
                    "to compute disjunctive variable bounds. False by default.",
        domain=bool
    ))
    return CONFIG


def _add_BB_configs(CONFIG):
    CONFIG.declare("check_sat", ConfigValue(
        default=False,
        domain=bool,
        description="When True, GDPopt-LBB will check satisfiability "
                    "at each node via the pyomo.contrib.satsolver interface"
    ))
    CONFIG.declare("solve_local_rnGDP", ConfigValue(
        default=False,
        domain=bool,
        description="When True, GDPopt-LBB will solve a local MINLP at each node."
    ))


def _add_subsolver_configs(CONFIG):
    CONFIG.declare("mip_solver", ConfigValue(
        default="gurobi",
        description="Mixed integer linear solver to use."
    ))
    CONFIG.declare("mip_solver_args", ConfigBlock(
        description="Keyword arguments to send to the MILP subsolver "
                    "solve() invocation",
        implicit=True))
    CONFIG.declare("nlp_solver", ConfigValue(
        default="ipopt",
        description="Nonlinear solver to use"))
    CONFIG.declare("nlp_solver_args", ConfigBlock(
        description="Keyword arguments to send to the NLP subsolver "
                    "solve() invocation",
        implicit=True))
    CONFIG.declare("minlp_solver", ConfigValue(
        default="baron",
        description="MINLP solver to use"
    ))
    CONFIG.declare("minlp_solver_args", ConfigBlock(
        description="Keyword arguments to send to the MINLP subsolver "
                    "solve() invocation",
        implicit=True))
    CONFIG.declare("local_minlp_solver", ConfigValue(
        default="bonmin",
        description="MINLP solver to use"
    ))
    CONFIG.declare("local_minlp_solver_args", ConfigBlock(
        description="Keyword arguments to send to the local MINLP subsolver "
                    "solve() invocation",
        implicit=True))


def _add_tolerance_configs(CONFIG):
    CONFIG.declare("bound_tolerance", ConfigValue(
        default=1E-6, domain=NonNegativeFloat,
        description="Tolerance for bound convergence."
    ))
    CONFIG.declare("small_dual_tolerance", ConfigValue(
        default=1E-8,
        description="When generating cuts, small duals multiplied "
                    "by expressions can cause problems. Exclude all duals "
                    "smaller in absolue value than the following."
    ))
    CONFIG.declare("integer_tolerance", ConfigValue(
        default=1E-5,
        description="Tolerance on integral values."
    ))
    CONFIG.declare("constraint_tolerance", ConfigValue(
        default=1E-6,
        description="Tolerance on constraint satisfaction."
    ))
    CONFIG.declare("variable_tolerance", ConfigValue(
        default=1E-8,
        description="Tolerance on variable bounds."
    ))
    CONFIG.declare("zero_tolerance", ConfigValue(
        default=1E-15,
        description="Tolerance on variable equal to zero."))
