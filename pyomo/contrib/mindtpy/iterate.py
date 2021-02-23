#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Iteration loop for MindtPy."""
from __future__ import division
import logging
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts

from pyomo.contrib.mindtpy.mip_solve import (solve_OA_master,
                                             handle_master_mip_optimal,
                                             handle_master_mip_other_conditions,
                                             handle_master_mip_infeasible)

from pyomo.contrib.mindtpy.nlp_solve import (solve_NLP_subproblem,
                                             handle_NLP_subproblem_optimal,
                                             handle_NLP_subproblem_infeasible,
                                             handle_NLP_subproblem_other_termination)

from pyomo.core import minimize, Objective, Var
from pyomo.opt.results import ProblemSense
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.gdpopt.util import get_main_elapsed_time
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt import SolverFactory

logger = logging.getLogger('pyomo.contrib.mindtpy')


def MindtPy_iteration_loop(solve_data, config):
    """
    Main loop for MindtPy Algorithms

    This is the outermost function for the algorithms in this package; this function controls the progression of
    solving the model.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    working_model = solve_data.working_model
    main_objective = next(
        working_model.component_data_objects(Objective, active=True))
    while solve_data.mip_iter < config.iteration_limit:

        config.logger.info(
            '---MindtPy Master Iteration %s---'
            % solve_data.mip_iter)

        solve_data.mip_subiter = 0
        # solve MILP master problem
        if config.strategy in {'OA', 'GOA', 'ECP'}:
            master_mip, master_mip_results = solve_OA_master(
                solve_data, config)
            if config.single_tree is False:
                if master_mip_results.solver.termination_condition is tc.optimal:
                    handle_master_mip_optimal(master_mip, solve_data, config)
                elif master_mip_results.solver.termination_condition is tc.infeasible:
                    handle_master_mip_infeasible(
                        master_mip, solve_data, config)
                    last_iter_cuts = True
                    break
                else:
                    handle_master_mip_other_conditions(master_mip, master_mip_results,
                                                       solve_data, config)
                # Call the MILP post-solve callback
                config.call_after_master_solve(master_mip, solve_data)
        else:
            raise NotImplementedError()

        if algorithm_should_terminate(solve_data, config, check_cycling=True):
            last_iter_cuts = False
            break

        if config.single_tree is False and config.strategy != 'ECP':  # if we don't use lazy callback, i.e. LP_NLP
            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            fixed_nlp, fixed_nlp_result = solve_NLP_subproblem(
                solve_data, config)
            if fixed_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                handle_NLP_subproblem_optimal(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition is tc.infeasible:
                handle_NLP_subproblem_infeasible(fixed_nlp, solve_data, config)
            else:
                handle_NLP_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                        solve_data, config)
            # Call the NLP post-solve callback
            config.call_after_subproblem_solve(fixed_nlp, solve_data)

        if algorithm_should_terminate(solve_data, config, check_cycling=False):
            last_iter_cuts = True
            break

        if config.strategy == 'ECP':
            add_ecp_cuts(solve_data.mip, solve_data, config)

        # if config.strategy == 'PSC':
        #     # If the hybrid algorithm is not making progress, switch to OA.
        #     progress_required = 1E-6
        #     if main_objective.sense == minimize:
        #         log = solve_data.LB_progress
        #         sign_adjust = 1
        #     else:
        #         log = solve_data.UB_progress
        #         sign_adjust = -1
        #     # Maximum number of iterations in which the lower (optimistic)
        #     # bound does not improve before switching to OA
        #     max_nonimprove_iter = 5
        #     making_progress = True
        #     # TODO-romeo Unneccesary for OA and LOA, right?
        #     for i in range(1, max_nonimprove_iter + 1):
        #         try:
        #             if (sign_adjust * log[-i]
        #                     <= (log[-i - 1] + progress_required)
        #                     * sign_adjust):
        #                 making_progress = False
        #             else:
        #                 making_progress = True
        #                 break
        #         except IndexError:
        #             # Not enough history yet, keep going.
        #             making_progress = True
        #             break
        #     if not making_progress and (
        #             config.strategy == 'hPSC' or
        #             config.strategy == 'PSC'):
        #         config.logger.info(
        #             'Not making enough progress for {} iterations. '
        #             'Switching to OA.'.format(max_nonimprove_iter))
        #         config.strategy = 'OA'

    # if add_nogood_cuts is True, the bound obtained in the last iteration is no reliable.
    # we correct it after the iteration.
    if config.add_nogood_cuts:
        bound_fix(solve_data, config, last_iter_cuts)


def algorithm_should_terminate(solve_data, config, check_cycling):
    """
    Checks if the algorithm should terminate at the given point

    This function determines whether the algorithm should terminate based on the solver options and progress.
    (Sets the solve_data.results.solver.termination_condition to the appropriate condition, i.e. optimal,
    maxIterations, maxTimeLimit)

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    check_cycling: bool
        check for a special case that causes a binary variable to loop through the same values

    Returns
    -------
    boolean
        True if the algorithm should terminate else returns False
    """

    # Check bound convergence
    if solve_data.LB + config.bound_tolerance >= solve_data.UB:
        config.logger.info(
            'MindtPy exiting on bound convergence. '
            'LB: {} + (tol {}) >= UB: {}\n'.format(
                solve_data.LB, config.bound_tolerance, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.optimal
        return True

    # Check iteration limit
    if solve_data.mip_iter >= config.iteration_limit:
        config.logger.info(
            'MindtPy unable to converge bounds '
            'after {} master iterations.'.format(solve_data.mip_iter))
        config.logger.info(
            'Final bound values: LB: {}  UB: {}'.
            format(solve_data.LB, solve_data.UB))
        if config.single_tree:
            solve_data.results.solver.termination_condition = tc.feasible
        else:
            solve_data.results.solver.termination_condition = tc.maxIterations
        return True

    # Check time limit
    if get_main_elapsed_time(solve_data.timing) >= config.time_limit:
        config.logger.info(
            'MindtPy unable to converge bounds '
            'before time limit of {} seconds. '
            'Elapsed: {} seconds'
            .format(config.time_limit, get_main_elapsed_time(solve_data.timing)))
        config.logger.info(
            'Final bound values: LB: {}  UB: {}'.
            format(solve_data.LB, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
        return True

    # Check if algorithm is stalling
    if len(solve_data.LB_progress) >= config.stalling_limit:
        if abs(solve_data.LB_progress[-1] - solve_data.LB_progress[-config.stalling_limit]) <= config.zero_tolerance:
            config.logger.info(
                'Algorithm is not making enough progress. '
                'Exiting iteration loop.')
            config.logger.info(
                'Final bound values: LB: {}  UB: {}'.
                format(solve_data.LB, solve_data.UB))
            if solve_data.best_solution_found is not None:
                solve_data.results.solver.termination_condition = tc.feasible
            else:
                solve_data.best_solution_found = solve_data.working_model.clone()
                config.logger.warning(
                    'Algorithm did not find a feasible solution. '
                    'Returning best bound solution. Consider increasing stalling_limit or bound_tolerance.')
                solve_data.results.solver.termination_condition = tc.noSolution

            return True

    if config.strategy == 'ECP':
        # check to see if the nonlinear constraints are satisfied
        MindtPy = solve_data.working_model.MindtPy_utils
        nonlinear_constraints = [c for c in MindtPy.constraint_list if
                                 c.body.polynomial_degree() not in (1, 0)]
        for nlc in nonlinear_constraints:
            if nlc.has_lb():
                try:
                    lower_slack = nlc.lslack()
                except (ValueError, OverflowError):
                    lower_slack = -10
                    # Use not fixed numbers in this case. Try some factor of ecp_tolerance
                if lower_slack < -config.ecp_tolerance:
                    config.logger.info(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(
                            nlc))
                    return False
            if nlc.has_ub():
                try:
                    upper_slack = nlc.uslack()
                except (ValueError, OverflowError):
                    upper_slack = -10
                if upper_slack < -config.ecp_tolerance:
                    config.logger.info(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(
                            nlc))
                    return False
        # For ECP to know whether to know which bound to copy over (primal or dual)
        if solve_data.objective_sense == 1:
            solve_data.UB = solve_data.LB
        else:
            solve_data.LB = solve_data.UB
        config.logger.info(
            'MindtPy-ECP exiting on nonlinear constraints satisfaction. '
            'LB: {} UB: {}\n'.format(
                solve_data.LB, solve_data.UB))

        solve_data.best_solution_found = solve_data.working_model.clone()
        solve_data.results.solver.termination_condition = tc.optimal
        return True
    # Cycling check
    if config.cycling_check and solve_data.mip_iter >= 1 and check_cycling:
        temp = []
        for var in solve_data.mip.component_data_objects(ctype=Var):
            if var.is_integer():
                temp.append(int(round(var.value)))
        solve_data.curr_int_sol = temp

        if solve_data.curr_int_sol == solve_data.prev_int_sol:
            config.logger.info(
                'Cycling happens after {} master iterations. '
                'This issue happens when the NLP subproblem violates constraint qualification. '
                'Convergence to optimal solution is not guaranteed.'
                .format(solve_data.mip_iter))
            config.logger.info(
                'Final bound values: LB: {}  UB: {}'.
                format(solve_data.LB, solve_data.UB))
            # TODO determine solve_data.LB, solve_data.UB is inf or -inf.
            solve_data.results.solver.termination_condition = tc.feasible
            return True

        solve_data.prev_int_sol = solve_data.curr_int_sol

    # if not algorithm_is_making_progress(solve_data, config):
    #     config.logger.debug(
    #         'Algorithm is not making enough progress. '
    #         'Exiting iteration loop.')
    #     return True
    return False


def bound_fix(solve_data, config, last_iter_cuts):
    if config.single_tree:
        config.logger.info(
            'Fix the bound to the value of one iteration before optimal solution is found.')
        if solve_data.results.problem.sense == ProblemSense.minimize:
            solve_data.LB = solve_data.stored_bound[solve_data.UB]
        else:
            solve_data.UB = solve_data.stored_bound[solve_data.LB]
    else:
        config.logger.info(
            'Solve the master problem without the last nogood cut to fix the bound.'
            'zero_tolerance is set to 1E-4')
        config.zero_tolerance = 1E-4
        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        if last_iter_cuts is False:
            fixed_nlp, fixed_nlp_result = solve_NLP_subproblem(
                solve_data, config)
            if fixed_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                handle_NLP_subproblem_optimal(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition is tc.infeasible:
                handle_NLP_subproblem_infeasible(fixed_nlp, solve_data, config)
            else:
                handle_NLP_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                        solve_data, config)

        MindtPy = solve_data.mip.MindtPy_utils
        # only deactivate the last integer cut.
        if config.strategy == 'GOA':
            if solve_data.results.problem.sense == ProblemSense.minimize:
                valid_no_good_cuts_num = solve_data.num_no_good_cuts_added[solve_data.UB]
            else:
                valid_no_good_cuts_num = solve_data.num_no_good_cuts_added[solve_data.LB]
            for i in range(valid_no_good_cuts_num+1, len(
                    MindtPy.MindtPy_linear_cuts.integer_cuts)+1):
                MindtPy.MindtPy_linear_cuts.integer_cuts[i].deactivate()
        elif config.strategy == 'OA':
            MindtPy.MindtPy_linear_cuts.integer_cuts[len(
                MindtPy.MindtPy_linear_cuts.integer_cuts)].deactivate()
        # MindtPy.MindtPy_linear_cuts.oa_cuts.activate()
        masteropt = SolverFactory(config.mip_solver)
        # determine if persistent solver is called.
        if isinstance(masteropt, PersistentSolver):
            masteropt.set_instance(solve_data.mip, symbolic_solver_labels=True)
        mip_args = dict(config.mip_solver_args)
        elapsed = get_main_elapsed_time(solve_data.timing)
        remaining = int(max(config.time_limit - elapsed, 1))
        if config.mip_solver == 'gams':
            mip_args['add_options'] = mip_args.get('add_options', [])
            mip_args['add_options'].append('option optcr=0.001;')
        if config.threads > 0:
            masteropt.options["threads"] = config.threads
        master_mip_results = masteropt.solve(
            solve_data.mip, tee=config.solver_tee, **mip_args)
        main_objective = next(
            solve_data.working_model.component_data_objects(Objective, active=True))
        if main_objective.sense == minimize:
            solve_data.LB = max(
                [master_mip_results.problem.lower_bound] + solve_data.LB_progress[:-1])
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                [master_mip_results.problem.upper_bound] + solve_data.UB_progress[:-1])
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'Fixed bound values: LB: {}  UB: {}'.
            format(solve_data.LB, solve_data.UB))
        # Check bound convergence
        if solve_data.LB + config.bound_tolerance >= solve_data.UB:
            solve_data.results.solver.termination_condition = tc.optimal
