"""Iteration loop for MindtPy."""
from __future__ import division

from pyomo.contrib.mindtpy.mip_solve import (
    solve_OA_master, handle_master_mip_optimal,
    handle_master_mip_other_conditions)
from pyomo.contrib.mindtpy.nlp_solve import (
    solve_NLP_subproblem, handle_NLP_subproblem_optimal,
    handle_NLP_subproblem_infeasible, handle_NLP_subproblem_other_termination)
from pyomo.core import minimize, Objective
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.gdpopt.util import (
    get_main_elapsed_time, copy_var_list_values)


def MindtPy_iteration_loop(solve_data, config):
    """Organizes the subcalls for decomposition methods

    In general, it runs
    - check termination
    - MIP master problem
      - result handling
    - check termination
    - NLP subproblem
    - repeat
    """

    while solve_data.mip_iter < config.iteration_limit:
        config.logger.info(
            '---MindtPy Master Iteration %s---'
            % solve_data.mip_iter)

        if algorithm_should_terminate(solve_data, config):
            break

        solve_data.mip_subiter = 0
        # solve MILP master problem
        if config.strategy == 'OA':
            master_mip, master_mip_results = solve_OA_master(solve_data, config)
            if master_mip_results.solver.termination_condition is tc.optimal:
                handle_master_mip_optimal(master_mip, solve_data, config)
            else:
                handle_master_mip_other_conditions(master_mip, master_mip_results,
                                                   solve_data, config)
            # Call the MILP post-solve callback
            config.call_after_master_solve(master_mip, solve_data)
        elif config.strategy == 'feas_pump':
            feas_mip, feas_mip_results = solve_OA_master(solve_data, config)
            if feas_mip_results.solver.termination_condition is tc.optimal:
                handle_master_mip_optimal(feas_mip, solve_data, config)
            elif feas_mip_results.solver.termination_condition is tc.infeasible:
                # This basically means the incumbent is the optimal solution
                if solve_data.best_solution_found is not None:
                    config.logger.info('Problem became infeasible. This means the feasibility pump has converged.')
                    solve_data.results.solver.termination_condition = tc.optimal
                else:
                    config.logger.info('No feasible solution has been found')
                    solve_data.results.solver.termination_condition = tc.infeasible
                break
            elif feas_mip_results.solver.termination_condition is tc.maxIterations:
                config.logger.error('No feasible solution has been found')
                solve_data.results.solver.termination_condition = tc.maxIterations
                break

        else:
            raise NotImplementedError()

        if algorithm_should_terminate(solve_data, config):
            break

        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        fix_nlp, fix_nlp_result = solve_NLP_subproblem(solve_data, config)
        if fix_nlp_result.solver.termination_condition is tc.optimal:
            handle_NLP_subproblem_optimal(fix_nlp, solve_data, config)
        elif fix_nlp_result.solver.termination_condition is tc.infeasible:
            handle_NLP_subproblem_infeasible(fix_nlp, solve_data, config)
        else:
            handle_NLP_subproblem_other_termination(fix_nlp, fix_nlp_result.solver.termination_condition,
                                                    solve_data, config)
        # Call the NLP post-solve callback
        config.call_after_subproblem_solve(fix_nlp, solve_data)

        if config.strategy == 'PSC':
            PSC_switch_to_OA_if_no_progress(solve_data, config)


def PSC_switch_to_OA_if_no_progress(solve_data, config):
    """If the hybrid algorithm is not making progress, switch to OA."""
    working_model = solve_data.working_model
    main_objective = next(working_model.component_data_objects(Objective, active=True))
    progress_required = 1E-6
    if main_objective.sense == minimize:
        log = solve_data.LB_progress
        sign_adjust = 1
    else:
        log = solve_data.UB_progress
        sign_adjust = -1
    # Maximum number of iterations in which the lower (optimistic)
    # bound does not improve before switching to OA
    max_nonimprove_iter = 5
    making_progress = True
    for i in range(1, max_nonimprove_iter + 1):
        try:
            if (sign_adjust * log[-i]
                    <= (log[-i - 1] + progress_required)
                    * sign_adjust):
                making_progress = False
            else:
                making_progress = True
                break
        except IndexError:
            # Not enough history yet, keep going.
            making_progress = True
            break
    if not making_progress and (
            config.strategy == 'hPSC' and
            config.strategy == 'PSC'):
        config.logger.info(
            'Not making enough progress for {} iterations. '
            'Switching to OA.'.format(max_nonimprove_iter))
        config.strategy = 'OA'


def algorithm_should_terminate(solve_data, config):
    """Check if the algorithm should terminate.

    Termination conditions based on solver options and progress.
    Sets the solve_data.results.solver.termination_condition to the appropriate
    condition, i.e. optimal, maxIterations, maxTimeLimit

    """
    if config.strategy in ['OA', 'LOA']:
        # Check bound convergence
        if solve_data.LB + config.bound_tolerance >= solve_data.UB:
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                'LB: {} + (tol {}) >= UB: {}\n'.format(
                    solve_data.LB, config.bound_tolerance, solve_data.UB))
            solve_data.results.solver.termination_condition = tc.optimal
            return True
    elif config.strategy is 'feas_pump':
        # feasability pump termination comes from infeasibility of the MIP
        pass

    # Check iteration limit
    if solve_data.mip_iter >= config.iteration_limit:
        config.logger.info(
            'MindtPy unable to converge bounds '
            'after {} master iterations.'.format(solve_data.mip_iter))
        config.logger.info(
            'Final bound values: LB: {}  UB: {}'.
            format(solve_data.LB, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.maxIterations
        return True

    # Check time limit
    if get_main_elapsed_time(solve_data.timing) > config.time_limit:
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
    # if not algorithm_is_making_progress(solve_data, config):
    #     config.logger.debug(
    #         'Algorithm is not making enough progress. '
    #         'Exiting iteration loop.')
    #     return True
    return False
