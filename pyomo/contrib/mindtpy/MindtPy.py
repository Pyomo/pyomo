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

from pyomo.contrib.gdpopt.util import (
    copy_var_list_values,
    create_utility_block,
    time_code,
    setup_results_object, process_objective, lower_logger_level_to)
from pyomo.contrib.mindtpy.initialization import MindtPy_initialize_master
from pyomo.contrib.mindtpy.iterate import MindtPy_iteration_loop
from pyomo.contrib.mindtpy.util import (
    MindtPySolveData, model_is_valid
)
from pyomo.core import (
    Block, ConstraintList, NonNegativeReals, Set, Suffix, Var,
    VarList, TransformationFactory, Objective)
from pyomo.opt import SolverFactory, SolverResults
from pyomo.common.collections import Container
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.mindtpy.config_options import _get_GDPopt_config

logger = logging.getLogger('pyomo.contrib.mindtpy')

__version__ = (0, 1, 0)


@SolverFactory.register(
    'mindtpy',
    doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')
class MindtPySolver(object):
    """A decomposition-based MINLP solver.
    """
    CONFIG = _get_GDPopt_config()

    def available(self, exception_flag=True):
        """Check if solver is available.
        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.
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
        Warning: at this point in time, if you try to use PSC or GBD with
        anything other than IPOPT as the NLP solver, bad things will happen.
        This is because the suffixes are not in place to extract dual values
        from the variable bounds for any other solver.
        TODO: fix needed with the GBD implementation.
        Args:
            model (Block): a Pyomo model or block to be solved
        """
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        # configuration confirmation
        if config.single_tree:
            config.iteration_limit = 1
            config.add_slack = False
            config.add_nogood_cuts = False
            config.mip_solver = 'cplex_persistent'
            config.logger.info(
                "Single tree implementation is activated. The defalt MIP solver is 'cplex_persistent'")
        # if the slacks fix to zero, just don't add them
        if config.max_slack == 0.0:
            config.add_slack = False

        if config.strategy == "GOA":
            config.add_nogood_cuts = True
            config.add_slack = True
            config.use_mcpp = True
            config.integer_to_binary = True
            config.use_dual = False
            config.use_fbbt = True

        if config.nlp_solver == "baron":
            config.use_dual = False
        # if ecp tolerance is not provided use bound tolerance
        if config.ecp_tolerance is None:
            config.ecp_tolerance = config.bound_tolerance

        # if the objective function is a constant, dual bound constraint is not added.
        obj = next(model.component_data_objects(ctype=Objective, active=True))
        if obj.expr.polynomial_degree() == 0:
            config.use_dual_bound = False

        solve_data = MindtPySolveData()
        solve_data.results = SolverResults()
        solve_data.timing = Container()
        solve_data.curr_int_sol = []
        solve_data.prev_int_sol = []

        if config.use_fbbt:
            fbbt(model)
            config.logger.info(
                "Use the fbbt to tighten the bounds of variables")

        solve_data.original_model = model
        solve_data.working_model = model.clone()
        if config.integer_to_binary:
            TransformationFactory('contrib.integer_to_binary'). \
                apply_to(solve_data.working_model)

        new_logging_level = logging.INFO if config.tee else None
        with time_code(solve_data.timing, 'total', is_main_timer=True), \
                lower_logger_level_to(config.logger, new_logging_level), \
                create_utility_block(solve_data.working_model, 'MindtPy_utils', solve_data):
            config.logger.info("---Starting MindtPy---")

            MindtPy = solve_data.working_model.MindtPy_utils
            setup_results_object(solve_data, config)
            process_objective(solve_data, config, use_mcpp=config.use_mcpp)

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
            feas = MindtPy.MindtPy_feas = Block()
            feas.deactivate()
            feas.feas_constraints = ConstraintList(
                doc='Feasibility Problem Constraints')

            # Create a model block in which to store the generated linear
            # constraints. Do not leave the constraints on by default.
            lin = MindtPy.MindtPy_linear_cuts = Block()
            lin.deactivate()

            # Integer cuts exclude particular discrete decisions
            lin.integer_cuts = ConstraintList(doc='integer cuts')
            # Feasible integer cuts exclude discrete realizations that have
            # been explored via an NLP subproblem. Depending on model
            # characteristics, the user may wish to revisit NLP subproblems
            # (with a different initialization, for example). Therefore, these
            # cuts are not enabled by default.
            #
            # Note: these cuts will only exclude integer realizations that are
            # not already in the primary integer_cuts ConstraintList.
            lin.feasible_integer_cuts = ConstraintList(
                doc='explored integer cuts')
            lin.feasible_integer_cuts.deactivate()

            # Set up iteration counters
            solve_data.nlp_iter = 0
            solve_data.mip_iter = 0
            solve_data.mip_subiter = 0

            # set up bounds
            solve_data.LB = float('-inf')
            solve_data.UB = float('inf')
            solve_data.LB_progress = [solve_data.LB]
            solve_data.UB_progress = [solve_data.UB]
            if config.single_tree and config.add_nogood_cuts:
                solve_data.stored_bound = {}
            if config.strategy == 'GOA' and config.add_nogood_cuts:
                solve_data.num_no_good_cuts_added = {}

            # Set of NLP iterations for which cuts were generated
            lin.nlp_iters = Set(dimen=1)

            # Set of MIP iterations for which cuts were generated in ECP
            lin.mip_iters = Set(dimen=1)

            if config.feasibility_norm == 'L1' or config.feasibility_norm == 'L2':
                feas.nl_constraint_set = Set(initialize=[i for i, constr in enumerate(MindtPy.constraint_list, 1) if
                                                         constr.body.polynomial_degree() not in (1, 0)],
                                             doc="Integer index set over the nonlinear constraints."
                                             "The set corresponds to the index of nonlinear constraint in constraint_set")
                # Create slack variables for feasibility problem
                feas.slack_var = Var(feas.nl_constraint_set,
                                     domain=NonNegativeReals, initialize=1)
            else:
                feas.slack_var = Var(domain=NonNegativeReals, initialize=1)

            # Create slack variables for OA cuts
            if config.add_slack:
                lin.slack_vars = VarList(
                    bounds=(0, config.max_slack), initialize=0, domain=NonNegativeReals)

            # Flag indicating whether the solution improved in the past
            # iteration or not
            solve_data.solution_improved = False

            if config.nlp_solver == 'ipopt':
                if not hasattr(solve_data.working_model, 'ipopt_zL_out'):
                    solve_data.working_model.ipopt_zL_out = Suffix(
                        direction=Suffix.IMPORT)
                if not hasattr(solve_data.working_model, 'ipopt_zU_out'):
                    solve_data.working_model.ipopt_zU_out = Suffix(
                        direction=Suffix.IMPORT)

            # Initialize the master problem
            with time_code(solve_data.timing, 'initialization'):
                MindtPy_initialize_master(solve_data, config)

            # Algorithm main loop
            with time_code(solve_data.timing, 'main loop'):
                MindtPy_iteration_loop(solve_data, config)

            if solve_data.best_solution_found is not None:
                # Update values in original model
                copy_var_list_values(
                    from_list=solve_data.best_solution_found.MindtPy_utils.variable_list,
                    to_list=MindtPy.variable_list,
                    config=config)
                # MindtPy.objective_value.set_value(
                #     value(solve_data.working_objective_expr, exception=False))
                copy_var_list_values(
                    MindtPy.variable_list,
                    solve_data.original_model.component_data_objects(Var),
                    config)

            solve_data.results.problem.lower_bound = solve_data.LB
            solve_data.results.problem.upper_bound = solve_data.UB

        solve_data.results.solver.timing = solve_data.timing
        solve_data.results.solver.user_time = solve_data.timing.total
        solve_data.results.solver.wallclock_time = solve_data.timing.total

        solve_data.results.solver.iterations = solve_data.mip_iter
        solve_data.results.solver.best_solution_found_time = solve_data.best_solution_found_time

        if config.single_tree:
            solve_data.results.solver.num_nodes = solve_data.nlp_iter - \
                (1 if config.init_strategy == 'rNLP' else 0)

        return solve_data.results

    #
    # Support "with" statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
