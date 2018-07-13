"""Iteration loop for MindtPy."""
from __future__ import division

from pyomo.contrib.mindtpy.cut_generation import add_ecp_cut
from pyomo.contrib.mindtpy.mip_solve import (solve_ECP_master,
                                             solve_GBD_master, solve_OA_master,
                                             solve_PSC_master)
from pyomo.contrib.mindtpy.nlp_solve import solve_NLP_subproblem
from pyomo.core import minimize


def MindtPy_iteration_loop(solve_data, config):
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
    while solve_data.mip_iter < config.iteration_limit:
        config.logger.info(
            '---MindtPy Master Iteration %s---'
            % solve_data.mip_iter)

        if algorithm_should_terminate(solve_data, config):
            break

        solve_data.mip_subiter = 0
        # solve MILP master problem
        if config.strategy == 'OA':
            solve_OA_master(solve_data, config)
        elif config.strategy == 'PSC':
            solve_PSC_master(solve_data, config)
        elif config.strategy == 'GBD':
            solve_GBD_master(solve_data, config)
        elif config.strategy == 'ECP':
            solve_ECP_master(solve_data, config)

        if algorithm_should_terminate(solve_data, config):
            break

        if config.strategy == 'ECP':
            # Add ECP cut
            add_ecp_cut(solve_data, config)
        else:
            # Solve NLP subproblem
            solve_NLP_subproblem(solve_data, config)

        # If the hybrid algorithm is not making progress, switch to OA.
        progress_required = 1E-6
        if MindtPy.objective.sense == minimize:
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
            print('Not making enough progress for {} iterations. '
                  'Switching to OA.'.format(max_nonimprove_iter))
            config.strategy = 'OA'


def algorithm_should_terminate(solve_data, config):
    """Check if the algorithm should terminate.

    Termination conditions based on solver options and progress.

    """
    # Check bound convergence
    if solve_data.LB + config.bound_tolerance >= solve_data.UB:
        config.logger.info(
            'MindtPy exiting on bound convergence. '
            'LB: {} + (tol {}) >= UB: {}\n'.format(
                solve_data.LB, config.bound_tolerance, solve_data.UB))
        # res.solver.termination_condition = tc.optimal
        return True

    # Check iteration limit
    if solve_data.mip_iter >= config.iteration_limit:
        config.logger.info(
            'MindtPy unable to converge bounds '
            'after {} master iterations.'.format(solve_data.mip_iter))
        config.logger.info(
            'Final bound values: LB: {}  UB: {}'.
            format(solve_data.LB, solve_data.UB))
        return True

    # if not algorithm_is_making_progress(solve_data, config):
    #     config.logger.debug(
    #         'Algorithm is not making enough progress. '
    #         'Exiting iteration loop.')
    #     return True
    return False
