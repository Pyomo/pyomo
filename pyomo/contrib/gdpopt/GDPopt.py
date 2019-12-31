# -*- coding: utf-8 -*-
"""Main driver module for GDPopt solver.

19.10.11 changes:
- bugfix on SolverStatus error message
19.5.13 changes:
- add handling to integer cuts for disjunct pruning during FBBT
19.4.23 changes:
- add support for linear subproblems
- use automatic differentiation for large constraints
- bugfixes on time limit support
- treat fixed variables as constants in GLOA cut generation
19.3.25 changes:
- add rudimentary time limit support
- start keeping basic changelog

"""
from __future__ import division

import logging

from pyomo.common.config import (
    ConfigBlock, ConfigList, ConfigValue, In, NonNegativeFloat, NonNegativeInt,
    add_docstring_list, PositiveInt
)
from pyomo.contrib.gdpopt.data_class import GDPoptSolveData
from pyomo.contrib.gdpopt.iterate import GDPopt_iteration_loop
from pyomo.contrib.gdpopt.master_initialize import (
    GDPopt_initialize_master, valid_init_strategies
)
from pyomo.contrib.gdpopt.util import (
    _DoNothing, a_logger, copy_var_list_values,
    create_utility_block, presolve_lp_nlp, process_objective,
    setup_results_object,
    restore_logger_level, time_code,
    setup_solver_environment)
from pyomo.core.base import ConstraintList
from pyomo.opt.base import SolverFactory
from pyomo.opt.results import SolverResults
from pyutilib.misc import Container

__version__ = (19, 5, 13)  # Note: date-based version number


@SolverFactory.register(
    'gdpopt',
    doc='The GDPopt decomposition-based '
    'Generalized Disjunctive Programming (GDP) solver')
class GDPoptSolver(object):
    """Decomposition solver for Generalized Disjunctive Programming (GDP) problems.

    The GDPopt (Generalized Disjunctive Programming optimizer) solver applies a
    variety of decomposition-based approaches to solve Generalized Disjunctive
    Programming (GDP) problems. GDP models can include nonlinear, continuous
    variables and constraints, as well as logical conditions.

    These approaches include:

    - Outer approximation
    - Partial surrogate cuts [pending]
    - Generalized Bender decomposition [pending]

    This solver implementation was developed by Carnegie Mellon University in the
    research group of Ignacio Grossmann.

    For nonconvex problems, the bounds self.LB and self.UB may not be rigorous.

    Questions: Please make a post at StackOverflow and/or contact Qi Chen
    <https://github.com/qtothec>.

    Keyword arguments below are specified for the :code:`solve` function.

    """

    def solve(self, model, **kwds):
        """Solve the model.

        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.

        This function performs all of the GDPopt solver setup and problem
        validation. It then calls upon helper functions to construct the
        initial master approximation and iteration loop.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        with setup_solver_environment(model, config) as solve_data:
            self._log_solver_intro_message(config)
            solve_data.results.solver.name = 'GDPopt %s - %s' % (
                str(self.version()), config.strategy)

            # Verify that objective has correct form
            process_objective(solve_data, config)

            # Presolve LP or NLP problems using subsolvers
            presolved, presolve_results = presolve_lp_nlp(solve_data, config)
            if presolved:
                # TODO merge the solver results
                return presolve_results  # problem presolved

            # Initialize the master problem
            with time_code(solve_data.timing, 'initialization'):
                GDPopt_initialize_master(solve_data, config)

            # Algorithm main loop
            with time_code(solve_data.timing, 'main loop'):
                GDPopt_iteration_loop(solve_data, config)

        return solve_data.results

    """Support use as a context manager under current solver API"""
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def _log_solver_intro_message(self, config):
        config.logger.info(
            "Starting GDPopt version %s using %s algorithm"
            % (".".join(map(str, self.version())), config.strategy)
        )
        config.logger.info(
            """
If you use this software, you may cite the following:
- Implementation:
Chen, Q; Johnson, ES; Siirola, JD; Grossmann, IE.
Pyomo.GDP: Disjunctive Models in Python. 
Proc. of the 13th Intl. Symposium on Process Systems Eng.
San Diego, 2018.
- LOA algorithm:
Türkay, M; Grossmann, IE.
Logic-based MINLP algorithms for the optimal synthesis of process networks.
Comp. and Chem. Eng. 1996, 20(8), 959–978.
DOI: 10.1016/0098-1354(95)00219-7.
- GLOA algorithm:
Lee, S; Grossmann, IE.
A Global Optimization Algorithm for Nonconvex Generalized Disjunctive Programming and Applications to Process Systems
Comp. and Chem. Eng. 2001, 25, 1675-1697.
DOI: 10.1016/S0098-1354(01)00732-3
        """.strip()
        )

    _metasolver = False

    CONFIG = ConfigBlock("GDPopt")
    CONFIG.declare("iterlim", ConfigValue(
        default=30, domain=NonNegativeInt,
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
        default="LOA", domain=In(["LOA", "GLOA"]),
        description="Decomposition strategy to use."
    ))
    CONFIG.declare("init_strategy", ConfigValue(
        default="set_covering", domain=In(valid_init_strategies.keys()),
        description="Initialization strategy to use.",
        doc="""Selects the initialization strategy to use when generating
        the initial cuts to construct the master problem."""
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
    CONFIG.declare("mip_solver", ConfigValue(
        default="gurobi",
        description="Mixed integer linear solver to use."
    ))
    CONFIG.declare("mip_presolve", ConfigValue(
        default=True,
        description="Flag to enable or diable Pyomo MIP presolve. Default=True.",
        domain=bool
    ))
    mip_solver_args = CONFIG.declare("mip_solver_args", ConfigBlock(
        description="Keyword arguments to send to the MILP subsolver "
        "solve() invocation",
        implicit=True))
    CONFIG.declare("nlp_solver", ConfigValue(
        default="ipopt",
        description="Nonlinear solver to use"))
    nlp_solver_args = CONFIG.declare("nlp_solver_args", ConfigBlock(
        description="Keyword arguments to send to the NLP subsolver "
        "solve() invocation",
        implicit=True))
    CONFIG.declare("subproblem_presolve", ConfigValue(
        default=True,
        description="Flag to enable or disable subproblem presolve. Default=True.",
        domain=bool
    ))
    CONFIG.declare("minlp_solver", ConfigValue(
        default="baron",
        description="MINLP solver to use"
    ))
    minlp_solver_args = CONFIG.declare("minlp_solver_args", ConfigBlock(
        description="Keyword arguments to send to the MINLP subsolver "
        "solve() invocation",
        implicit=True))
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
    CONFIG.declare("round_discrete_vars", ConfigValue(
        default=True,
        description="flag to round subproblem discrete variable values to the nearest integer. "
        "Rounding is done before fixing disjuncts."
    ))
    CONFIG.declare("force_subproblem_nlp", ConfigValue(
        default=False,
        description="Force subproblems to be NLP, even if discrete variables exist."
    ))


# Add the CONFIG arguments to the solve method docstring
GDPoptSolver.solve.__doc__ = add_docstring_list(
    GDPoptSolver.solve.__doc__, GDPoptSolver.CONFIG, indent_spacing=4)
