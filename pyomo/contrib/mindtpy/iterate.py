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
from pyomo.contrib.mindtpy.util import set_solver_options, get_integer_solution, update_suboptimal_dual_bound, copy_var_list_values_from_solution_pool
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts

from pyomo.contrib.mindtpy.mip_solve import solve_main, handle_main_optimal, handle_main_infeasible, handle_main_other_conditions, handle_regularization_main_tc
from pyomo.contrib.mindtpy.nlp_solve import solve_subproblem, handle_nlp_subproblem_tc
from pyomo.core import minimize, maximize
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt import SolverFactory
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.gdpopt.util import copy_var_list_values
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from operator import itemgetter

tabu_list, tabu_list_available = attempt_import(
    'pyomo.contrib.mindtpy.tabu_list')


def MindtPy_iteration_loop(solve_data, config):
    """Main loop for MindtPy Algorithms.

    This is the outermost function for the algorithms in this package; this function controls the progression of
    solving the model.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.

    Raises
    ------
    ValueError
        The strategy value is not correct or not included.
    """
    last_iter_cuts = False
    while solve_data.mip_iter < config.iteration_limit:

        solve_data.mip_subiter = 0
        # solve MILP main problem
        if config.strategy in {'OA', 'GOA', 'ECP'}:
            main_mip, main_mip_results = solve_main(solve_data, config)
            if main_mip_results is not None:
                if not config.single_tree:
                    if main_mip_results.solver.termination_condition is tc.optimal:
                        handle_main_optimal(main_mip, solve_data, config)
                    elif main_mip_results.solver.termination_condition is tc.infeasible:
                        handle_main_infeasible(main_mip, solve_data, config)
                        last_iter_cuts = True
                        break
                    else:
                        handle_main_other_conditions(
                            main_mip, main_mip_results, solve_data, config)
                    # Call the MILP post-solve callback
                    with time_code(solve_data.timing, 'Call after main solve'):
                        config.call_after_main_solve(main_mip, solve_data)
            else:
                config.logger.info('Algorithm should terminate here.')
                break
        else:
            raise ValueError()

        # Regularization is activated after the first feasible solution is found.
        if config.add_regularization is not None and solve_data.best_solution_found is not None and not config.single_tree:
            # The main problem might be unbounded, regularization is activated only when a valid bound is provided.
            if solve_data.dual_bound != solve_data.dual_bound_progress[0]:
                main_mip, main_mip_results = solve_main(
                    solve_data, config, regularization_problem=True)
                handle_regularization_main_tc(
                    main_mip, main_mip_results, solve_data, config)

        # TODO: add descriptions for the following code
        if config.add_regularization is not None and config.single_tree:
            solve_data.curr_int_sol = get_integer_solution(
                solve_data.mip, string_zero=True)
            copy_var_list_values(
                main_mip.MindtPy_utils.variable_list,
                solve_data.working_model.MindtPy_utils.variable_list,
                config)
            if solve_data.curr_int_sol not in set(solve_data.integer_list):
                fixed_nlp, fixed_nlp_result = solve_subproblem(
                    solve_data, config)
                handle_nlp_subproblem_tc(
                    fixed_nlp, fixed_nlp_result, solve_data, config)
        if algorithm_should_terminate(solve_data, config, check_cycling=True):
            last_iter_cuts = False
            break

        if not config.single_tree and config.strategy != 'ECP':  # if we don't use lazy callback, i.e. LP_NLP
            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            if not config.solution_pool:
                fixed_nlp, fixed_nlp_result = solve_subproblem(
                    solve_data, config)
                handle_nlp_subproblem_tc(
                    fixed_nlp, fixed_nlp_result, solve_data, config)

                # Call the NLP post-solve callback
                with time_code(solve_data.timing, 'Call after subproblem solve'):
                    config.call_after_subproblem_solve(fixed_nlp, solve_data)

                if algorithm_should_terminate(solve_data, config, check_cycling=False):
                    last_iter_cuts = True
                    break
            else:
                if config.mip_solver == 'cplex_persistent':
                    solution_pool_names = main_mip_results._solver_model.solution.pool.get_names()
                elif config.mip_solver == 'gurobi_persistent':
                    solution_pool_names = list(
                        range(main_mip_results._solver_model.SolCount))
                # list to store the name and objective value of the solutions in the solution pool
                solution_name_obj = []
                for name in solution_pool_names:
                    if config.mip_solver == 'cplex_persistent':
                        obj = main_mip_results._solver_model.solution.pool.get_objective_value(
                            name)
                    elif config.mip_solver == 'gurobi_persistent':
                        main_mip_results._solver_model.setParam(
                            gurobipy.GRB.Param.SolutionNumber, name)
                        obj = main_mip_results._solver_model.PoolObjVal
                    solution_name_obj.append([name, obj])
                solution_name_obj.sort(
                    key=itemgetter(1), reverse=solve_data.objective_sense == maximize)
                counter = 0
                for name, _ in solution_name_obj:
                    # the optimal solution of the main problem has been added to integer_list above
                    # so we should skip checking cycling for the first solution in the solution pool
                    if counter >= 1:
                        copy_var_list_values_from_solution_pool(solve_data.mip.MindtPy_utils.variable_list,
                                                                solve_data.working_model.MindtPy_utils.variable_list,
                                                                config, solver_model=main_mip_results._solver_model,
                                                                var_map=main_mip_results._pyomo_var_to_solver_var_map,
                                                                solution_name=name)
                        solve_data.curr_int_sol = get_integer_solution(
                            solve_data.working_model)
                        if solve_data.curr_int_sol in set(solve_data.integer_list):
                            config.logger.info(
                                'The same combination has been explored and will be skipped here.')
                            continue
                        else:
                            solve_data.integer_list.append(
                                solve_data.curr_int_sol)
                    counter += 1
                    fixed_nlp, fixed_nlp_result = solve_subproblem(
                        solve_data, config)
                    handle_nlp_subproblem_tc(
                        fixed_nlp, fixed_nlp_result, solve_data, config)

                    # Call the NLP post-solve callback
                    with time_code(solve_data.timing, 'Call after subproblem solve'):
                        config.call_after_subproblem_solve(
                            fixed_nlp, solve_data)

                    if algorithm_should_terminate(solve_data, config, check_cycling=False):
                        last_iter_cuts = True
                        break

                    if counter >= config.num_solution_iteration:
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
        #     # TODO-romeo Unnecessary for OA and ROA, right?
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
    if (config.add_no_good_cuts or config.use_tabu_list) and config.strategy != 'FP' and not solve_data.should_terminate and config.add_regularization is None:
        fix_dual_bound(solve_data, config, last_iter_cuts)
    config.logger.info(
        ' ===============================================================================================')


def algorithm_should_terminate(solve_data, config, check_cycling):
    """Checks if the algorithm should terminate at the given point.

    This function determines whether the algorithm should terminate based on the solver options and progress.
    (Sets the solve_data.results.solver.termination_condition to the appropriate condition, i.e. optimal,
    maxIterations, maxTimeLimit).

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    check_cycling : bool
        Whether to check for a special case that causes the discrete variables to loop through the same values.

    Returns
    -------
    bool
        True if the algorithm should terminate, False otherwise.
    """
    if solve_data.should_terminate:
        if solve_data.primal_bound == solve_data.primal_bound_progress[0]:
            solve_data.results.solver.termination_condition = tc.noSolution
        else:
            solve_data.results.solver.termination_condition = tc.feasible
        return True

    # Check bound convergence
    if solve_data.abs_gap <= config.absolute_bound_tolerance:
        config.logger.info(
            'MindtPy exiting on bound convergence. '
            '|Primal Bound: {} - Dual Bound: {}| <= (absolute tolerance {})  \n'.format(
                solve_data.primal_bound, solve_data.dual_bound, config.absolute_bound_tolerance))
        solve_data.results.solver.termination_condition = tc.optimal
        return True
    # Check relative bound convergence
    if solve_data.best_solution_found is not None:
        if solve_data.rel_gap <= config.relative_bound_tolerance:
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                '|Primal Bound: {} - Dual Bound: {}| / (1e-10 + |Primal Bound|:{}) <= relative tolerance: {}'.format(solve_data.primal_bound, solve_data.dual_bound, abs(solve_data.primal_bound), config.relative_bound_tolerance))
            solve_data.results.solver.termination_condition = tc.optimal
            return True

    # Check iteration limit
    if solve_data.mip_iter >= config.iteration_limit:
        config.logger.info(
            'MindtPy unable to converge bounds '
            'after {} main iterations.'.format(solve_data.mip_iter))
        config.logger.info(
            'Final bound values: Primal Bound: {}  Dual Bound: {}'.
            format(solve_data.primal_bound, solve_data.dual_bound))
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
            'Final bound values: Primal Bound: {}  Dual Bound: {}'.
            format(solve_data.primal_bound, solve_data.dual_bound))
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
        return True

    # Check if algorithm is stalling
    if len(solve_data.primal_bound_progress) >= config.stalling_limit:
        if abs(solve_data.primal_bound_progress[-1] - solve_data.primal_bound_progress[-config.stalling_limit]) <= config.zero_tolerance:
            config.logger.info(
                'Algorithm is not making enough progress. '
                'Exiting iteration loop.')
            config.logger.info(
                'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                format(solve_data.primal_bound, solve_data.dual_bound))
            if solve_data.best_solution_found is not None:
                solve_data.results.solver.termination_condition = tc.feasible
            else:
                # TODO: Is it correct to set solve_data.working_model as the best_solution_found?
                # In function copy_var_list_values, skip_fixed is set to True in default.
                solve_data.best_solution_found = solve_data.working_model.clone()
                config.logger.warning(
                    'Algorithm did not find a feasible solution. '
                    'Returning best bound solution. Consider increasing stalling_limit or absolute_bound_tolerance.')
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
                    # Set lower_slack (upper_slack below) less than -config.ecp_tolerance in this case.
                    lower_slack = -10*config.ecp_tolerance
                if lower_slack < -config.ecp_tolerance:
                    config.logger.debug(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(nlc))
                    return False
            if nlc.has_ub():
                try:
                    upper_slack = nlc.uslack()
                except (ValueError, OverflowError):
                    upper_slack = -10*config.ecp_tolerance
                if upper_slack < -config.ecp_tolerance:
                    config.logger.debug(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(nlc))
                    return False
        # For ECP to know whether to know which bound to copy over (primal or dual)
        solve_data.primal_bound = solve_data.dual_bound
        config.logger.info(
            'MindtPy-ECP exiting on nonlinear constraints satisfaction. '
            'Primal Bound: {} Dual Bound: {}\n'.format(solve_data.primal_bound, solve_data.dual_bound))

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
                        'Cycling happens after {} main iterations. '
                        'The same combination is obtained in iteration {} '
                        'This issue happens when the NLP subproblem violates constraint qualification. '
                        'Convergence to optimal solution is not guaranteed.'
                        .format(solve_data.mip_iter, solve_data.integer_list.index(solve_data.curr_int_sol)+1))
                    config.logger.info(
                        'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                        format(solve_data.primal_bound, solve_data.dual_bound))
                    # TODO determine solve_data.primal_bound, solve_data.dual_bound is inf or -inf.
                    solve_data.results.solver.termination_condition = tc.feasible
                    return True
            solve_data.integer_list.append(solve_data.curr_int_sol)

    # if not algorithm_is_making_progress(solve_data, config):
    #     config.logger.debug(
    #         'Algorithm is not making enough progress. '
    #         'Exiting iteration loop.')
    #     return True
    return False


def fix_dual_bound(solve_data, config, last_iter_cuts):
    """Fix the dual bound when no-good cuts or tabu list is activated.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    last_iter_cuts : bool
        Whether the cuts in the last iteration have been added.
    """
    if config.single_tree:
        config.logger.info(
            'Fix the bound to the value of one iteration before optimal solution is found.')
        try:
            solve_data.dual_bound = solve_data.stored_bound[solve_data.primal_bound]
        except KeyError:
            config.logger.info('No stored bound found. Bound fix failed.')
    else:
        config.logger.info(
            'Solve the main problem without the last no_good cut to fix the bound.'
            'zero_tolerance is set to 1E-4')
        config.zero_tolerance = 1E-4
        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        if not last_iter_cuts:
            fixed_nlp, fixed_nlp_result = solve_subproblem(solve_data, config)
            handle_nlp_subproblem_tc(
                fixed_nlp, fixed_nlp_result, solve_data, config)

        MindtPy = solve_data.mip.MindtPy_utils
        # deactivate the integer cuts generated after the best solution was found.
        if config.strategy == 'GOA':
            try:
                valid_no_good_cuts_num = solve_data.num_no_good_cuts_added[solve_data.primal_bound]
                if config.add_no_good_cuts:
                    for i in range(valid_no_good_cuts_num+1, len(MindtPy.cuts.no_good_cuts)+1):
                        MindtPy.cuts.no_good_cuts[i].deactivate()
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
        mainopt = SolverFactory(config.mip_solver)
        # determine if persistent solver is called.
        if isinstance(mainopt, PersistentSolver):
            mainopt.set_instance(solve_data.mip, symbolic_solver_labels=True)
        if config.use_tabu_list:
            tabulist = mainopt._solver_model.register_callback(
                tabu_list.IncumbentCallback_cplex)
            tabulist.solve_data = solve_data
            tabulist.opt = mainopt
            tabulist.config = config
            mainopt._solver_model.parameters.preprocessing.reduce.set(1)
            # If the callback is used to reject incumbents, the user must set the
            # parameter c.parameters.preprocessing.reduce either to the value 1 (one)
            # to restrict presolve to primal reductions only or to 0 (zero) to disable all presolve reductions
            mainopt._solver_model.set_warning_stream(None)
            mainopt._solver_model.set_log_stream(None)
            mainopt._solver_model.set_error_stream(None)
        mip_args = dict(config.mip_solver_args)
        set_solver_options(mainopt, solve_data, config, solver_type='mip')
        main_mip_results = mainopt.solve(
            solve_data.mip, tee=config.mip_solver_tee, **mip_args)
        if main_mip_results.solver.termination_condition is tc.infeasible:
            config.logger.info(
                'Bound fix failed. The bound fix problem is infeasible')
        else:
            update_suboptimal_dual_bound(solve_data, main_mip_results)
            config.logger.info(
                'Fixed bound values: Primal Bound: {}  Dual Bound: {}'.
                format(solve_data.primal_bound, solve_data.dual_bound))
        # Check bound convergence
        if abs(solve_data.primal_bound - solve_data.dual_bound) <= config.absolute_bound_tolerance:
            solve_data.results.solver.termination_condition = tc.optimal
