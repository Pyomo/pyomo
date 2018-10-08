"""Iteration code."""
from __future__ import division

from pyomo.contrib.gdpopt.cut_generation import (add_integer_cut,
                                                 add_outer_approximation_cuts,
                                                 add_affine_cuts)
from pyomo.contrib.gdpopt.mip_solve import solve_GLOA_master, solve_LOA_master
from pyomo.contrib.gdpopt.nlp_solve import (solve_global_NLP,
                                            solve_LOA_subproblem)
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.gdpopt.util import time_code


def GDPopt_iteration_loop(solve_data, config):
    """Algorithm main loop.

    Returns True if successful convergence is obtained. False otherwise.

    """
    while solve_data.master_iteration < config.iterlim:
        # Set iteration counters for new master iteration.
        solve_data.master_iteration += 1
        solve_data.mip_iteration = 0
        solve_data.nlp_iteration = 0

        # print line for visual display
        config.logger.info(
            '---GDPopt Master Iteration %s---'
            % solve_data.master_iteration)

        # solve linear master problem
        with time_code(solve_data.timing, 'mip'):
            if solve_data.current_strategy == 'LOA':
                mip_result = solve_LOA_master(solve_data, config)
            elif solve_data.current_strategy == 'GLOA':
                mip_result = solve_GLOA_master(solve_data, config)

        # Check termination conditions
        if algorithm_should_terminate(solve_data, config):
            break

        # Solve NLP subproblem
        if solve_data.current_strategy == 'LOA':
            with time_code(solve_data.timing, 'nlp'):
                nlp_result = solve_LOA_subproblem(
                    mip_result.var_values, solve_data, config)
            if nlp_result.feasible:
                add_outer_approximation_cuts(nlp_result, solve_data, config)
        elif solve_data.current_strategy == 'GLOA':
            with time_code(solve_data.timing, 'nlp'):
                nlp_result = solve_global_NLP(
                    mip_result.var_values, solve_data, config)
            if nlp_result.feasible:
                add_affine_cuts(nlp_result, solve_data, config)

        # Add integer cut
        add_integer_cut(
            mip_result.var_values, solve_data, config,
            feasible=nlp_result.feasible)

        # Check termination conditions
        if algorithm_should_terminate(solve_data, config):
            break


def algorithm_should_terminate(solve_data, config):
    """Check if the algorithm should terminate.

    Termination conditions based on solver options and progress.

    """
    # Check bound convergence
    if solve_data.LB + config.bound_tolerance >= solve_data.UB:
        config.logger.info(
            'GDPopt exiting on bound convergence. '
            'LB: %s + (tol %s) >= UB: %s' %
            (solve_data.LB, config.bound_tolerance,
             solve_data.UB))
        solve_data.results.solver.termination_condition = tc.optimal
        return True

    # Check iteration limit
    if solve_data.master_iteration >= config.iterlim:
        config.logger.info(
            'GDPopt unable to converge bounds '
            'after %s master iterations.'
            % (solve_data.master_iteration,))
        config.logger.info(
            'Final bound values: LB: %s  UB: %s'
            % (solve_data.LB, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.maxIterations
        return True

    if not algorithm_is_making_progress(solve_data, config):
        config.logger.debug(
            'Algorithm is not making enough progress. '
            'Exiting iteration loop.')
        solve_data.results.solver.termination_condition = tc.locallyOptimal
        return True
    return False


def algorithm_is_making_progress(solve_data, config):
    """Make sure that the algorithm is making sufficient progress
    at each iteration to continue."""

    # TODO if backtracking is turned on, and algorithm visits the same point
    # twice without improvement in objective value, turn off backtracking.

    # TODO stop iterations if feasible solutions not progressing for a number
    # of iterations.

    # If the hybrid algorithm is not making progress, switch to OA.
    # required_feas_prog = 1E-6
    # if solve_data.working_model.GDPopt_utils.objective.sense == minimize:
    #     sign_adjust = 1
    # else:
    #     sign_adjust = -1

    # Maximum number of iterations in which feasible bound does not
    # improve before terminating algorithm
    # if (len(feas_prog_log) > config.algorithm_stall_after and
    #     (sign_adjust * (feas_prog_log[-1] + required_feas_prog)
    #      >= sign_adjust *
    #      feas_prog_log[-1 - config.algorithm_stall_after])):
    #     config.logger.info(
    #         'Feasible solutions not making enough progress '
    #         'for %s iterations. Algorithm stalled. Exiting.\n'
    #         'To continue, increase value of parameter '
    #         'algorithm_stall_after.'
    #         % (config.algorithm_stall_after,))
    #     return False

    return True
