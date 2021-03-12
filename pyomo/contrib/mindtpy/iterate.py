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
from pyomo.contrib.mindtpy.util import set_solver_options, get_integer_solution
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts

from pyomo.contrib.mindtpy.mip_solve import (solve_master,
                                             handle_master_optimal, handle_master_infeasible, handle_master_other_conditions)
from pyomo.contrib.mindtpy.nlp_solve import (solve_subproblem,
                                             handle_subproblem_optimal, handle_subproblem_infeasible,
                                             handle_subproblem_other_termination)
from pyomo.core import minimize, maximize, Var
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt import SolverFactory
from pyomo.common.dependencies import attempt_import

tabu_list, tabu_list_available = attempt_import(
    'pyomo.contrib.mindtpy.tabu_list')

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
    last_iter_cuts = False
    while solve_data.mip_iter < config.iteration_limit:

        config.logger.info(
            '---MindtPy Master Iteration %s---'
            % (solve_data.mip_iter+1))

        solve_data.mip_subiter = 0
        # solve MILP master problem
        if config.strategy in {'OA', 'GOA', 'ECP'}:
            master_mip, master_mip_results = solve_master(
                solve_data, config)
            if master_mip_results is not None:
                if config.single_tree is False:
                    if master_mip_results.solver.termination_condition is tc.optimal:
                        handle_master_optimal(master_mip, solve_data, config)
                    elif master_mip_results.solver.termination_condition is tc.infeasible:
                        handle_master_infeasible(
                            master_mip, solve_data, config)
                        last_iter_cuts = True
                        break
                    else:
                        handle_master_other_conditions(master_mip, master_mip_results,
                                                       solve_data, config)
                    # Call the MILP post-solve callback
                    with time_code(solve_data.timing, 'Call after master solve'):
                        config.call_after_master_solve(master_mip, solve_data)
            else:
                config.logger.info('Algorithm should terminate here.')
                break
        else:
            raise NotImplementedError()

        # regularization is activated after the first feasible solution is found.
        if config.add_regularization is not None and solve_data.best_solution_found is not None and config.single_tree is False:
            # the master problem might be unbounded, regularization is activated only when a valid bound is provided.
            if (solve_data.objective_sense == minimize and solve_data.LB != float('-inf')) or (solve_data.objective_sense == maximize and solve_data.UB != float('inf')):
                master_mip, master_mip_results = solve_master(
                    solve_data, config, regularization_problem=True)
                if master_mip_results is None:
                    config.logger.info(
                        'Failed to solve the projection problem.'
                        'The solution of the OA master problem will be adopted.')
                elif master_mip_results.solver.termination_condition in {tc.optimal, tc.feasible}:
                    handle_master_optimal(
                        master_mip, solve_data, config, update_bound=False)
                elif master_mip_results.solver.termination_condition is tc.maxTimeLimit:
                    config.logger.info(
                        'Regularization problem failed to converge within the time limit.')
                    solve_data.results.solver.termination_condition = tc.maxTimeLimit
                    break
                elif master_mip_results.solver.termination_condition is tc.infeasible:
                    config.logger.info(
                        'Regularization problem infeasible.')
                elif master_mip_results.solver.termination_condition is tc.unbounded:
                    config.logger.info(
                        'Regularization problem ubounded.'
                        'Sometimes solving MIQP in cplex, unbounded means infeasible.')
                elif master_mip_results.solver.termination_condition is tc.unknown:
                    config.logger.info(
                        'Termination condition of the projection problem is unknown.')
                    if master_mip_results.problem.lower_bound != float('-inf'):
                        config.logger.info('Solution limit has been reached.')
                        handle_master_optimal(
                            master_mip, solve_data, config, update_bound=False)
                    else:
                        config.logger.info('No solution obtained from the projection subproblem.'
                                           'Please set mip_solver_tee to True for more informations.'
                                           'The solution of the OA master problem will be adopted.')
                else:
                    raise ValueError(
                        'MindtPy unable to handle projection problem termination condition '
                        'of %s. Solver message: %s' %
                        (master_mip_results.solver.termination_condition, master_mip_results.solver.message))

        if algorithm_should_terminate(solve_data, config, check_cycling=True):
            last_iter_cuts = False
            break

        if config.single_tree is False and config.strategy != 'ECP':  # if we don't use lazy callback, i.e. LP_NLP
            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            fixed_nlp, fixed_nlp_result = solve_subproblem(
                solve_data, config)
            if fixed_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                handle_subproblem_optimal(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
                handle_subproblem_infeasible(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition is tc.maxTimeLimit:
                config.logger.info(
                    'NLP subproblem failed to converge within the time limit.')
                solve_data.results.solver.termination_condition = tc.maxTimeLimit
                break
            elif fixed_nlp_result.solver.termination_condition is tc.maxEvaluations:
                config.logger.info(
                    'NLP subproblem failed due to maxEvaluations.')
                solve_data.results.solver.termination_condition = tc.maxEvaluations
                break
            else:
                handle_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                    solve_data, config)
            # Call the NLP post-solve callback
            with time_code(solve_data.timing, 'Call after subproblem solve'):
                config.call_after_subproblem_solve(fixed_nlp, solve_data)

        if algorithm_should_terminate(solve_data, config, check_cycling=False):
            last_iter_cuts = True
            break

        if config.strategy == 'ECP':
            add_ecp_cuts(solve_data.mip, solve_data, config)

        # if config.strategy == 'PSC':
        #     # If the hybrid algorithm is not making progress, switch to OA.
        #     progress_required = 1E-6
        #     if solve_data.objective_sense == minimize:
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

    # if add_no_good_cuts is True, the bound obtained in the last iteration is no reliable.
    # we correct it after the iteration.
    if (config.add_no_good_cuts or config.use_tabu_list) and config.strategy is not 'FP' and not solve_data.should_terminate and config.add_regularization is None:
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
    if solve_data.should_terminate:
        if solve_data.objective_sense == minimize:
            if solve_data.UB == float('inf'):
                solve_data.results.solver.termination_condition = tc.noSolution
            else:
                solve_data.results.solver.termination_condition = tc.feasible
        else:
            if solve_data.LB == float('-inf'):
                solve_data.results.solver.termination_condition = tc.noSolution
            else:
                solve_data.results.solver.termination_condition = tc.feasible
        return True

    # Check bound convergence
    if solve_data.LB + config.bound_tolerance >= solve_data.UB:
        config.logger.info(
            'MindtPy exiting on bound convergence. '
            'LB: {} + (tol {}) >= UB: {}\n'.format(
                solve_data.LB, config.bound_tolerance, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.optimal
        return True
    # Check relative bound convergence
    if solve_data.best_solution_found is not None:
        if solve_data.UB - solve_data.LB <= config.relative_bound_tolerance * (abs(solve_data.UB if solve_data.objective_sense == minimize else solve_data.LB) + 1E-10):
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                '(UB: {} - LB: {})/ (1e-10+|bestinteger|:{}) <= relative tolerance: {}'.format(solve_data.UB, solve_data.LB, abs(solve_data.UB if solve_data.objective_sense == minimize else solve_data.LB), config.relative_bound_tolerance))
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
                # TODO: Is it correct to set solve_data.working_model as the best_solution_found?
                # In function copy_var_list_values, skip_fixed is set to True in default.
                solve_data.best_solution_found = solve_data.working_model.clone()
                config.logger.warning(
                    'Algorithm did not find a feasible solution. '
                    'Returning best bound solution. Consider increasing stalling_limit or bound_tolerance.')
                solve_data.results.solver.termination_condition = tc.noSolution

            return True

    if config.strategy == 'ECP':
        # check to see if the nonlinear constraints are satisfied
        MindtPy = solve_data.working_model.MindtPy_utils
        nonlinear_constraints = [c for c in MindtPy.nonlinear_constraint_list]
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
        if solve_data.objective_sense == minimize:
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
    if check_cycling:
        if config.cycling_check or config.use_tabu_list:
            solve_data.curr_int_sol = get_integer_solution(solve_data.mip)
            if config.cycling_check and solve_data.mip_iter >= 1:
                if solve_data.curr_int_sol in set(solve_data.integer_list):
                    config.logger.info(
                        'Cycling happens after {} master iterations. '
                        'The same combination is obtained in iteration {} '
                        'This issue happens when the NLP subproblem violates constraint qualification. '
                        'Convergence to optimal solution is not guaranteed.'
                        .format(solve_data.mip_iter, solve_data.integer_list.index(solve_data.curr_int_sol)+1))
                    config.logger.info(
                        'Final bound values: LB: {}  UB: {}'.
                        format(solve_data.LB, solve_data.UB))
                    # TODO determine solve_data.LB, solve_data.UB is inf or -inf.
                    solve_data.results.solver.termination_condition = tc.feasible
                    return True
            solve_data.integer_list.append(solve_data.curr_int_sol)

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
        try:
            if solve_data.objective_sense == minimize:
                solve_data.LB = solve_data.stored_bound[solve_data.UB]
            else:
                solve_data.UB = solve_data.stored_bound[solve_data.LB]
        except KeyError:
            config.logger.info('No stored bound found. Bound fix failed.')
    else:
        config.logger.info(
            'Solve the master problem without the last no_good cut to fix the bound.'
            'zero_tolerance is set to 1E-4')
        config.zero_tolerance = 1E-4
        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        if last_iter_cuts is False:
            fixed_nlp, fixed_nlp_result = solve_subproblem(
                solve_data, config)
            if fixed_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                handle_subproblem_optimal(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
                handle_subproblem_infeasible(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition is tc.maxTimeLimit:
                config.logger.info(
                    'NLP subproblem failed to converge within the time limit.')
                solve_data.results.solver.termination_condition = tc.maxTimeLimit
            elif fixed_nlp_result.solver.termination_condition is tc.maxEvaluations:
                config.logger.info(
                    'NLP subproblem failed due to maxEvaluations.')
                solve_data.results.solver.termination_condition = tc.maxEvaluations
            else:
                handle_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                    solve_data, config)

        MindtPy = solve_data.mip.MindtPy_utils
        # deactivate the integer cuts generated after the best solution was found.
        if config.strategy == 'GOA':
            try:
                if solve_data.objective_sense == minimize:
                    valid_no_good_cuts_num = solve_data.num_no_good_cuts_added[solve_data.UB]
                else:
                    valid_no_good_cuts_num = solve_data.num_no_good_cuts_added[solve_data.LB]
                if config.add_no_good_cuts:
                    for i in range(valid_no_good_cuts_num+1, len(
                            MindtPy.cuts.no_good_cuts)+1):
                        MindtPy.cuts.no_good_cuts[i].deactivate(
                        )
                if config.use_tabu_list:
                    solve_data.integer_list = solve_data.integer_list[:valid_no_good_cuts_num]
            except KeyError:
                config.logger.info('No-good cut deactivate failed.')
        elif config.strategy == 'OA':
            # Only deactive the last OA cuts may not be correct.
            # Since integer solution may also be cut off by OA cuts due to calculation approximation.
            if config.add_no_good_cuts:
                MindtPy.cuts.no_good_cuts[len(
                    MindtPy.cuts.no_good_cuts)].deactivate()
            if config.use_tabu_list:
                solve_data.integer_list = solve_data.integer_list[:-1]
        if config.add_regularization is not None and MindtPy.find_component('mip_obj') is None:
            MindtPy.objective_list[-1].activate()
        masteropt = SolverFactory(config.mip_solver)
        # determine if persistent solver is called.
        if isinstance(masteropt, PersistentSolver):
            masteropt.set_instance(solve_data.mip, symbolic_solver_labels=True)
        if config.use_tabu_list:
            tabulist = masteropt._solver_model.register_callback(
                tabu_list.IncumbentCallback_cplex)
            tabulist.solve_data = solve_data
            tabulist.opt = masteropt
            tabulist.config = config
            masteropt._solver_model.parameters.preprocessing.reduce.set(1)
            # If the callback is used to reject incumbents, the user must set the
            # parameter c.parameters.preprocessing.reduce either to the value 1 (one)
            # to restrict presolve to primal reductions only or to 0 (zero) to disable all presolve reductions
            masteropt._solver_model.set_warning_stream(None)
            masteropt._solver_model.set_log_stream(None)
            masteropt._solver_model.set_error_stream(None)
        mip_args = dict(config.mip_solver_args)
        set_solver_options(masteropt, solve_data, config, solver_type='mip')
        master_mip_results = masteropt.solve(
            solve_data.mip, tee=config.mip_solver_tee, **mip_args)
        if master_mip_results.solver.termination_condition is tc.infeasible:
            config.logger.info(
                'Bound fix failed. The bound fix problem is infeasible')
        else:
            if solve_data.objective_sense == minimize:
                solve_data.LB = max(
                    [master_mip_results.problem.lower_bound] + solve_data.LB_progress[:-1])
                solve_data.bound_improved = solve_data.LB > solve_data.LB_progress[-1]
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(
                    [master_mip_results.problem.upper_bound] + solve_data.UB_progress[:-1])
                solve_data.bound_improved = solve_data.UB < solve_data.UB_progress[-1]
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'Fixed bound values: LB: {}  UB: {}'.
                format(solve_data.LB, solve_data.UB))
        # Check bound convergence
        if solve_data.LB + config.bound_tolerance >= solve_data.UB:
            solve_data.results.solver.termination_condition = tc.optimal
