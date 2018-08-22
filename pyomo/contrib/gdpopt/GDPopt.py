# -*- coding: utf-8 -*-
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

"""
from __future__ import division

import logging

import pyomo.common.plugin
from pyomo.common.config import (ConfigBlock, ConfigList, ConfigValue, In,
                                 NonNegativeFloat, NonNegativeInt)
from pyomo.contrib.gdpopt.data_class import GDPoptSolveData
from pyomo.contrib.gdpopt.iterate import GDPopt_iteration_loop
from pyomo.contrib.gdpopt.master_initialize import (GDPopt_initialize_master,
                                                    valid_init_strategies)
from pyomo.contrib.gdpopt.util import (_DoNothing, a_logger,
                                       build_ordered_component_lists,
                                       clone_orig_model_with_lists,
                                       copy_var_list_values,
                                       create_utility_block, model_is_valid,
                                       process_objective,
                                       record_original_model_statistics,
                                       record_working_model_statistics,
                                       reformulate_integer_variables,
                                       restore_logger_level, time_code)
from pyomo.core.base import ConstraintList, value
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt.base import IOptSolver
from pyomo.opt.results import SolverResults
from pyutilib.misc import Container

__version__ = (0, 4, 1)


class GDPoptSolver(pyomo.common.plugin.Plugin):
    """A decomposition-based GDP solver."""

    pyomo.common.plugin.implements(IOptSolver)
    pyomo.common.plugin.alias(
        'gdpopt',
        doc='The GDPopt decomposition-based '
        'Generalized Disjunctive Programming (GDP) solver')

    _metasolver = False

    CONFIG = ConfigBlock("GDPopt")
    CONFIG.declare("iterlim", ConfigValue(
        default=30, domain=NonNegativeInt,
        description="Iteration limit."
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
    mip_solver_args = CONFIG.declare(
        "mip_solver_args", ConfigBlock(implicit=True))
    CONFIG.declare("nlp_solver", ConfigValue(
        default="ipopt",
        description="Nonlinear solver to use"))
    nlp_solver_args = CONFIG.declare(
        "nlp_solver_args", ConfigBlock(implicit=True))
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
    CONFIG.declare("round_NLP_binaries", ConfigValue(
        default=True,
        description="flag to round binary values to exactly 0 or 1. "
        "Rounding is done before fixing disjuncts."
    ))
    CONFIG.declare("reformulate_integer_vars_using", ConfigValue(
        default=None,
        description="The method to use for reformulating integer variables "
        "into binary for this solver."
    ))

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

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
        solve_data = GDPoptSolveData()
        solve_data.results = SolverResults()
        solve_data.timing = Container()

        old_logger_level = config.logger.getEffectiveLevel()
        with time_code(solve_data.timing, 'total'), \
                restore_logger_level(config.logger), \
                create_utility_block(model, 'GDPopt_utils'):
            if config.tee and old_logger_level > logging.INFO:
                # If the logger does not already include INFO, include it.
                config.logger.setLevel(logging.INFO)
            config.logger.info("---Starting GDPopt---")

            solve_data.original_model = model

            solve_data.working_model = clone_orig_model_with_lists(model)
            GDPopt = solve_data.working_model.GDPopt_utils
            record_original_model_statistics(solve_data, config)

            solve_data.current_strategy = config.strategy

            # Reformulate integer variables to binary
            reformulate_integer_variables(solve_data.working_model, config)
            process_objective(solve_data, config)

            # Save ordered lists of main modeling components, so that data can
            # be easily transferred between future model clones.
            build_ordered_component_lists(solve_data.working_model)
            record_working_model_statistics(solve_data, config)
            solve_data.results.solver.name = 'GDPopt ' + str(self.version())

            # Save model initial values. These are used later to initialize NLP
            # subproblems.
            solve_data.initial_var_values = list(
                v.value for v in GDPopt.working_var_list)

            # Store the initial model state as the best solution found. If we
            # find no better solution, then we will restore from this copy.
            solve_data.best_solution_found = solve_data.initial_var_values

            # Validate the model to ensure that GDPopt is able to solve it.
            if not model_is_valid(solve_data, config):
                return

            # Maps in order to keep track of certain generated constraints
            GDPopt.oa_cut_map = ComponentMap()

            # Integer cuts exclude particular discrete decisions
            GDPopt.integer_cuts = ConstraintList(doc='integer cuts')

            # Feasible integer cuts exclude discrete realizations that have
            # been explored via an NLP subproblem. Depending on model
            # characteristics, the user may wish to revisit NLP subproblems
            # (with a different initialization, for example). Therefore, these
            # cuts are not enabled by default, unless the initial model has no
            # discrete decisions.

            # Note: these cuts will only exclude integer realizations that are
            # not already in the primary GDPopt_integer_cuts ConstraintList.
            GDPopt.no_backtracking = ConstraintList(
                doc='explored integer cuts')

            # Set up iteration counters
            solve_data.master_iteration = 0
            solve_data.mip_iteration = 0
            solve_data.nlp_iteration = 0

            # set up bounds
            solve_data.LB = float('-inf')
            solve_data.UB = float('inf')
            solve_data.iteration_log = {}

            # Flag indicating whether the solution improved in the past
            # iteration or not
            solve_data.feasible_solution_improved = False

            # Initialize the master problem
            with time_code(solve_data.timing, 'initialization'):
                GDPopt_initialize_master(solve_data, config)

            # Algorithm main loop
            with time_code(solve_data.timing, 'main loop'):
                GDPopt_iteration_loop(solve_data, config)

            # Update values in working model
            copy_var_list_values(
                from_list=solve_data.best_solution_found,
                to_list=GDPopt.working_var_list,
                config=config)
            GDPopt.objective_value.set_value(
                value(solve_data.working_objective_expr, exception=False))

            # Update values in original model
            copy_var_list_values(
                GDPopt.orig_var_list,
                solve_data.original_model.GDPopt_utils.orig_var_list,
                config)

            solve_data.results.problem.lower_bound = solve_data.LB
            solve_data.results.problem.upper_bound = solve_data.UB

        solve_data.results.solver.timing = solve_data.timing

        return solve_data.results
