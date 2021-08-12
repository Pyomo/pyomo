# -*- coding: utf-8 -*-

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Implementation of the MindtPy solver.

The MindtPy (MINLP Decomposition Toolkit) solver applies a variety of
decomposition-based approaches to solve nonlinear continuous-discrete problems.
These approaches include:

- Outer approximation
- Benders decomposition [pending]
- Partial surrogate cuts [pending]
- Extended cutting plane [pending]

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.

For nonconvex problems, the bounds solve_data.LB and solve_data.UB may not be
rigorous. Questions: Please make a post at StackOverflow and/or David Bernal
<https://github.com/bernalde>

"""
from __future__ import division
import logging
from pyomo.contrib.gdpopt.util import (copy_var_list_values, create_utility_block,
                                       time_code, setup_results_object, process_objective, lower_logger_level_to)
from pyomo.contrib.mindtpy.initialization import MindtPy_initialize_main
from pyomo.contrib.mindtpy.iterate import MindtPy_iteration_loop
from pyomo.contrib.mindtpy.util import model_is_valid, setup_solve_data
from pyomo.core import (Block, ConstraintList, NonNegativeReals,
                        Set, Suffix, Var, VarList, TransformationFactory, Objective, RangeSet)
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_config, check_config

logger = logging.getLogger('pyomo.contrib.mindtpy')

__version__ = (0, 1, 0)


@SolverFactory.register(
    'mindtpy',
    doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')
class MindtPySolver(object):
    """A decomposition-based MINLP solver.
    """
    CONFIG = _get_MindtPy_config()

    def available(self, exception_flag=True):
        """Check if solver is available.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def solve(self, model, **kwds):
        """Solve the model.

        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.

        Args:
            model (Block): a Pyomo model or block to be solved
        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        check_config(config)

        solve_data = setup_solve_data(model, config)

        if config.integer_to_binary:
            TransformationFactory('contrib.integer_to_binary'). \
                apply_to(solve_data.working_model)

        new_logging_level = logging.INFO if config.tee else None
        with time_code(solve_data.timing, 'total', is_main_timer=True), \
                lower_logger_level_to(config.logger, new_logging_level), \
                create_utility_block(solve_data.working_model, 'MindtPy_utils', solve_data):
            config.logger.info('---Starting MindtPy---')

            MindtPy = solve_data.working_model.MindtPy_utils
            setup_results_object(solve_data, config)
            process_objective(solve_data, config,
                              move_linear_objective=(config.init_strategy == 'FP'
                                                     or config.add_regularization is not None),
                              use_mcpp=config.use_mcpp,
                              updata_var_con_list=config.add_regularization is None
                              )
            # The epigraph constraint is very "flat" for branching rules,
            # we want to use to original model for the main mip.
            if MindtPy.objective_list[0].expr.polynomial_degree() in {1, 0} and config.add_regularization is not None:
                MindtPy.objective_list[0].activate()
                MindtPy.objective_constr.deactivate()
                MindtPy.objective.deactivate()

            # Save model initial values.
            solve_data.initial_var_values = list(
                v.value for v in MindtPy.variable_list)

            # Store the initial model state as the best solution found. If we
            # find no better solution, then we will restore from this copy.
            solve_data.best_solution_found = None
            solve_data.best_solution_found_time = None

            # Record solver name
            solve_data.results.solver.name = 'MindtPy' + str(config.strategy)

            # Validate the model to ensure that MindtPy is able to solve it.
            if not model_is_valid(solve_data, config):
                return

            # Create a model block in which to store the generated feasibility
            # slack constraints. Do not leave the constraints on by default.
            feas = MindtPy.feas_opt = Block()
            feas.deactivate()
            feas.feas_constraints = ConstraintList(
                doc='Feasibility Problem Constraints')

            # Create a model block in which to store the generated linear
            # constraints. Do not leave the constraints on by default.
            lin = MindtPy.cuts = Block()
            lin.deactivate()

            # no-good cuts exclude particular discrete decisions
            lin.no_good_cuts = ConstraintList(doc='no-good cuts')
            # Feasible no-good cuts exclude discrete realizations that have
            # been explored via an NLP subproblem. Depending on model
            # characteristics, the user may wish to revisit NLP subproblems
            # (with a different initialization, for example). Therefore, these
            # cuts are not enabled by default.
            #
            # Note: these cuts will only exclude integer realizations that are
            # not already in the primary no_good_cuts ConstraintList.
            lin.feasible_no_good_cuts = ConstraintList(
                doc='explored no-good cuts')
            lin.feasible_no_good_cuts.deactivate()

            if config.feasibility_norm == 'L1' or config.feasibility_norm == 'L2':
                feas.nl_constraint_set = RangeSet(len(MindtPy.nonlinear_constraint_list),
                                                  doc='Integer index set over the nonlinear constraints.')
                # Create slack variables for feasibility problem
                feas.slack_var = Var(feas.nl_constraint_set,
                                     domain=NonNegativeReals, initialize=1)
            else:
                feas.slack_var = Var(domain=NonNegativeReals, initialize=1)

            # Create slack variables for OA cuts
            if config.add_slack:
                lin.slack_vars = VarList(
                    bounds=(0, config.max_slack), initialize=0, domain=NonNegativeReals)

            # Initialize the main problem
            with time_code(solve_data.timing, 'initialization'):
                MindtPy_initialize_main(solve_data, config)

            # Algorithm main loop
            with time_code(solve_data.timing, 'main loop'):
                MindtPy_iteration_loop(solve_data, config)
            if solve_data.best_solution_found is not None:
                # Update values in original model
                copy_var_list_values(
                    from_list=solve_data.best_solution_found.MindtPy_utils.variable_list,
                    to_list=MindtPy.variable_list,
                    config=config)
                copy_var_list_values(
                    MindtPy.variable_list,
                    [i for i in solve_data.original_model.component_data_objects(
                        Var) if not i.fixed],
                    config)
                # exclude fixed variables here. This is consistent with the definition of variable_list in GDPopt.util

            solve_data.results.problem.lower_bound = solve_data.LB
            solve_data.results.problem.upper_bound = solve_data.UB

        solve_data.results.solver.timing = solve_data.timing
        solve_data.results.solver.user_time = solve_data.timing.total
        solve_data.results.solver.wallclock_time = solve_data.timing.total
        solve_data.results.solver.iterations = solve_data.mip_iter
        solve_data.results.solver.num_infeasible_nlp_subproblem = solve_data.nlp_infeasible_counter
        solve_data.results.solver.best_solution_found_time = solve_data.best_solution_found_time

        if config.single_tree:
            solve_data.results.solver.num_nodes = solve_data.nlp_iter - \
                (1 if config.init_strategy == 'rNLP' else 0)

        return solve_data.results

    #
    # Support 'with' statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
