#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

def bounds_converged(solve_info, config):
    if solve_info.LB + config.bound_tolerance >= solve_info.UB:
        config.logger.info(
            'GDPopt exiting on bound convergence. '
            'LB: {:.10g} + (tol {:.10g}) >= UB: {:.10g}'.format(
                solve_info.LB, config.bound_tolerance, solve_info.UB))
        if solve_info.LB == float('inf') and solve_info.UB == float('inf'):
            solve_info.results.solver.termination_condition = tc.infeasible
        elif solve_info.LB == float('-inf') and solve_info.UB == float('-inf'):
            solve_info.results.solver.termination_condition = tc.infeasible
        else:
            solve_info.results.solver.termination_condition = tc.optimal
        return True
    return False

def reached_iteration_limit(solve_info, config):
    if solve_data.master_iteration >= config.iterlim:
        config.logger.info(
            'GDPopt unable to converge bounds '
            'after %s master iterations.'
            % (solve_data.master_iteration,))
        config.logger.info(
            'Final bound values: LB: {:.10g}  UB: {:.10g}'.format(
                solve_data.LB, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.maxIterations
        return True
    return False

def reached_time_limit(solve_info, config):
    elapsed = get_main_elapsed_time(solve_data.timing)
    if elapsed >= config.time_limit:
        config.logger.info(
            'GDPopt unable to converge bounds '
            'before time limit of {} seconds. '
            'Elapsed: {} seconds'
            .format(config.time_limit, elapsed))
        config.logger.info(
            'Final bound values: LB: {}  UB: {}'.
            format(solve_data.LB, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
        return True
    return False

def any_termination_criterion_met(solve_info, config):
    return (bounds_converged(solve_info, config) or 
            reached_iteration_limit(solve_info, config) or 
            reached_time_limit(solve_info_config))
