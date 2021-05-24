#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""main problem functions."""
from __future__ import division
import logging
from pyomo.core import Constraint, Expression, Objective, minimize, value
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.contrib.gdpopt.util import copy_var_list_values, SuppressInfeasibleWarning, _DoNothing, get_main_elapsed_time, time_code
from pyomo.contrib.gdpopt.mip_solve import distinguish_mip_infeasible_or_unbounded
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.mindtpy.util import generate_norm1_objective_function, generate_norm2sq_objective_function, generate_norm_inf_objective_function, generate_lag_objective_function, set_solver_options
logger = logging.getLogger('pyomo.contrib.mindtpy')

single_tree, single_tree_available = attempt_import(
    'pyomo.contrib.mindtpy.single_tree')
tabu_list, tabu_list_available = attempt_import(
    'pyomo.contrib.mindtpy.tabu_list')


def solve_main(solve_data, config, fp=False, regularization_problem=False):
    """
    This function solves the MIP main problem

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    Returns
    -------
    solve_data.mip: Pyomo model
        the MIP stored in solve_data
    main_mip_results: Pyomo results object
        result from solving the main MIP
    fp: Bool
        generate the feasibility pump regularization main problem
    regularization_problem: Bool
        generate the ROA regularization main problem
    """
    if fp:
        config.logger.info('FP-MIP %s: Solve main problem.' %
                           (solve_data.fp_iter,))
    elif regularization_problem:
        config.logger.info('Regularization-MIP %s: Solve main regularization problem.' %
                           (solve_data.mip_iter,))
    else:
        solve_data.mip_iter += 1
        config.logger.info('MIP %s: Solve main problem.' %
                           (solve_data.mip_iter,))

    # setup main problem
    setup_main(solve_data, config, fp, regularization_problem)
    mainopt = setup_mip_solver(solve_data, config, regularization_problem)

    mip_args = dict(config.mip_solver_args)
    if config.mip_solver in {'cplex', 'cplex_persistent', 'gurobi', 'gurobi_persistent'}:
        mip_args['warmstart'] = True
    set_solver_options(mainopt, solve_data, config,
                       solver_type='mip', regularization=regularization_problem)
    try:
        with time_code(solve_data.timing, 'regularization main' if regularization_problem else ('fp main' if fp else 'main')):
            main_mip_results = mainopt.solve(solve_data.mip,
                                             tee=config.mip_solver_tee, **mip_args)
    except (ValueError, AttributeError):
        if config.single_tree:
            config.logger.warning('Single tree terminate.')
            if get_main_elapsed_time(solve_data.timing) >= config.time_limit - 2:
                config.logger.warning('due to the timelimit.')
                solve_data.results.solver.termination_condition = tc.maxTimeLimit
            if config.strategy == 'GOA' or config.add_no_good_cuts:
                config.logger.warning('ValueError: Cannot load a SolverResults object with bad status: error. '
                                      'MIP solver failed. This usually happens in the single-tree GOA algorithm. '
                                      "No-good cuts are added and GOA algorithm doesn't converge within the time limit. "
                                      'No integer solution is found, so the cplex solver will report an error status. ')
        return None, None
    if main_mip_results.solver.termination_condition is tc.optimal:
        if config.single_tree and not config.add_no_good_cuts and not regularization_problem:
            if solve_data.objective_sense == minimize:
                solve_data.LB = max(
                    main_mip_results.problem.lower_bound, solve_data.LB)
                solve_data.bound_improved = solve_data.LB > solve_data.LB_progress[-1]
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(
                    main_mip_results.problem.upper_bound, solve_data.UB)
                solve_data.bound_improved = solve_data.UB < solve_data.UB_progress[-1]
                solve_data.UB_progress.append(solve_data.UB)

    elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(
            solve_data.mip, config)
        return solve_data.mip, main_mip_results

    if regularization_problem:
        solve_data.mip.MindtPy_utils.objective_constr.deactivate()
        solve_data.mip.MindtPy_utils.del_component('loa_proj_mip_obj')
        solve_data.mip.MindtPy_utils.cuts.del_component('obj_reg_estimate')
        if config.add_regularization == 'level_L1':
            solve_data.mip.MindtPy_utils.del_component('L1_obj')
        elif config.add_regularization == 'level_L_infinity':
            solve_data.mip.MindtPy_utils.del_component(
                'L_infinity_obj')

    return solve_data.mip, main_mip_results


def setup_mip_solver(solve_data, config, regularization_problem):
    # Deactivate extraneous IMPORT/EXPORT suffixes
    if config.nlp_solver == 'ipopt':
        getattr(solve_data.mip, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(solve_data.mip, 'ipopt_zU_out', _DoNothing()).deactivate()
    if regularization_problem:
        mainopt = SolverFactory(config.mip_regularization_solver)
    else:
        mainopt = SolverFactory(config.mip_solver)
    # determine if persistent solver is called.
    if isinstance(mainopt, PersistentSolver):
        mainopt.set_instance(solve_data.mip, symbolic_solver_labels=True)
    if config.single_tree and not regularization_problem:
        # Configuration of lazy callback
        lazyoa = mainopt._solver_model.register_callback(
            single_tree.LazyOACallback_cplex)
        # pass necessary data and parameters to lazyoa
        lazyoa.main_mip = solve_data.mip
        lazyoa.solve_data = solve_data
        lazyoa.config = config
        lazyoa.opt = mainopt
        mainopt._solver_model.set_warning_stream(None)
        mainopt._solver_model.set_log_stream(None)
        mainopt._solver_model.set_error_stream(None)
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
    return mainopt


# The following functions deal with handling the solution we get from the above MIP solver function


def handle_main_optimal(main_mip, solve_data, config, update_bound=True):
    """
    This function copies the result from 'solve_main' to the working model and updates the upper/lower bound. This
    function is called after an optimal solution is found for the main problem.

    Parameters
    ----------
    main_mip: Pyomo model
        the MIP main problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # proceed. Just need integer values
    MindtPy = main_mip.MindtPy_utils
    # check if the value of binary variable is valid
    for var in MindtPy.discrete_variable_list:
        if var.value is None:
            config.logger.warning(
                "Integer variable {} not initialized. It is set to it's lower bound".format(var.name))
            var.value = var.lb  # nlp_var.bounds[0]
    # warm start for the nlp subproblem
    copy_var_list_values(
        main_mip.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)

    if update_bound:
        if solve_data.objective_sense == minimize:
            solve_data.LB = max(
                value(MindtPy.mip_obj.expr), solve_data.LB)
            solve_data.bound_improved = solve_data.LB > solve_data.LB_progress[-1]
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                value(MindtPy.mip_obj.expr), solve_data.UB)
            solve_data.bound_improved = solve_data.UB < solve_data.UB_progress[-1]
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s  TIME: %ss'
            % (solve_data.mip_iter, value(MindtPy.mip_obj.expr),
               solve_data.LB, solve_data.UB, round(get_main_elapsed_time(solve_data.timing), 2)))


def handle_main_other_conditions(main_mip, main_mip_results, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem (given any of a few
    edge conditions, such as if the solution is neither infeasible nor optimal).

    Parameters
    ----------
    main_mip: Pyomo model
        the MIP main problem
    main_mip_results: Pyomo results object
        result from solving the MIP problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    if main_mip_results.solver.termination_condition is tc.infeasible:
        handle_main_infeasible(main_mip, solve_data, config)
    elif main_mip_results.solver.termination_condition is tc.unbounded:
        temp_results = handle_main_unbounded(main_mip, solve_data, config)
    elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        temp_results = handle_main_unbounded(main_mip, solve_data, config)
        if temp_results.solver.termination_condition is tc.infeasible:
            handle_main_infeasible(main_mip, solve_data, config)
    elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
        handle_main_max_timelimit(
            main_mip, main_mip_results, solve_data, config)
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
    elif (main_mip_results.solver.termination_condition is tc.other and
          main_mip_results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        MindtPy = main_mip.MindtPy_utils
        config.logger.info(
            'MILP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
        copy_var_list_values(
            main_mip.MindtPy_utils.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
        if solve_data.objective_sense == minimize:
            solve_data.LB = max(
                main_mip_results.problem.lower_bound, solve_data.LB)
            solve_data.bound_improved = solve_data.LB > solve_data.LB_progress[-1]
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                main_mip_results.problem.upper_bound, solve_data.UB)
            solve_data.bound_improved = solve_data.UB < solve_data.UB_progress[-1]
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.mip_obj.expr),
               solve_data.LB, solve_data.UB))
    else:
        raise ValueError(
            'MindtPy unable to handle MILP main termination condition '
            'of %s. Solver message: %s' %
            (main_mip_results.solver.termination_condition, main_mip_results.solver.message))


def handle_main_infeasible(main_mip, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem given an infeasible solution.

    Parameters
    ----------
    main_mip: Pyomo model
        the MIP main problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    config.logger.info(
        'MILP main problem is infeasible. '
        'Problem may have no more feasible '
        'binary configurations.')
    if solve_data.mip_iter == 1:
        config.logger.warning(
            'MindtPy initialization may have generated poor '
            'quality cuts.')
    # TODO no-good cuts for single tree case
    # set optimistic bound to infinity
    if solve_data.objective_sense == minimize:
        solve_data.LB_progress.append(solve_data.LB)
    else:
        solve_data.UB_progress.append(solve_data.UB)
    config.logger.info(
        'MindtPy exiting due to MILP main problem infeasibility.')
    if solve_data.results.solver.termination_condition is None:
        if solve_data.mip_iter == 0:
            solve_data.results.solver.termination_condition = tc.infeasible
        else:
            solve_data.results.solver.termination_condition = tc.feasible


def handle_main_max_timelimit(main_mip, main_mip_results, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem given that solving the
    MIP takes too long.

    Parameters
    ----------
    main_mip: Pyomo model
        the MIP main problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # TODO check that status is actually ok and everything is feasible
    MindtPy = main_mip.MindtPy_utils
    config.logger.info(
        'Unable to optimize MILP main problem '
        'within time limit. '
        'Using current solver feasible solution.')
    copy_var_list_values(
        main_mip.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)
    if solve_data.objective_sense == minimize:
        solve_data.LB = max(
            main_mip_results.problem.lower_bound, solve_data.LB)
        solve_data.bound_improved = solve_data.LB > solve_data.LB_progress[-1]
        solve_data.LB_progress.append(solve_data.LB)
    else:
        solve_data.UB = min(
            main_mip_results.problem.upper_bound, solve_data.UB)
        solve_data.bound_improved = solve_data.UB < solve_data.UB_progress[-1]
        solve_data.UB_progress.append(solve_data.UB)
    config.logger.info(
        'MIP %s: OBJ: %s  LB: %s  UB: %s'
        % (solve_data.mip_iter, value(MindtPy.mip_obj.expr),
           solve_data.LB, solve_data.UB))


def handle_main_unbounded(main_mip, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem given an unbounded solution
    due to the relaxation.

    Parameters
    ----------
    main_mip: Pyomo model
        the MIP main problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # Solution is unbounded. Add an arbitrary bound to the objective and resolve.
    # This occurs when the objective is nonlinear. The nonlinear objective is moved
    # to the constraints, and deactivated for the linear main problem.
    MindtPy = main_mip.MindtPy_utils
    config.logger.warning(
        'main MILP was unbounded. '
        'Resolving with arbitrary bound values of (-{0:.10g}, {0:.10g}) on the objective. '
        'You can change this bound with the option obj_bound.'.format(config.obj_bound))
    MindtPy.objective_bound = Constraint(
        expr=(-config.obj_bound, MindtPy.mip_obj.expr, config.obj_bound))
    mainopt = SolverFactory(config.mip_solver)
    if isinstance(mainopt, PersistentSolver):
        mainopt.set_instance(main_mip)
    set_solver_options(mainopt, solve_data, config, solver_type='mip')
    with SuppressInfeasibleWarning():
        main_mip_results = mainopt.solve(
            main_mip, tee=config.mip_solver_tee, **config.mip_solver_args)
    return main_mip_results


def handle_regularization_main_tc(main_mip, main_mip_results, solve_data, config):
    if main_mip_results is None:
        config.logger.info(
            'Failed to solve the regularization problem.'
            'The solution of the OA main problem will be adopted.')
    elif main_mip_results.solver.termination_condition in {tc.optimal, tc.feasible}:
        handle_main_optimal(
            main_mip, solve_data, config, update_bound=False)
    elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
        config.logger.info(
            'Regularization problem failed to converge within the time limit.')
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
        # break
    elif main_mip_results.solver.termination_condition is tc.infeasible:
        config.logger.info(
            'Regularization problem infeasible.')
    elif main_mip_results.solver.termination_condition is tc.unbounded:
        config.logger.info(
            'Regularization problem ubounded.'
            'Sometimes solving MIQP in cplex, unbounded means infeasible.')
    elif main_mip_results.solver.termination_condition is tc.unknown:
        config.logger.info(
            'Termination condition of the regularization problem is unknown.')
        if main_mip_results.problem.lower_bound != float('-inf'):
            config.logger.info('Solution limit has been reached.')
            handle_main_optimal(
                main_mip, solve_data, config, update_bound=False)
        else:
            config.logger.info('No solution obtained from the regularization subproblem.'
                               'Please set mip_solver_tee to True for more informations.'
                               'The solution of the OA main problem will be adopted.')
    else:
        raise ValueError(
            'MindtPy unable to handle regularization problem termination condition '
            'of %s. Solver message: %s' %
            (main_mip_results.solver.termination_condition, main_mip_results.solver.message))


def setup_main(solve_data, config, fp, regularization_problem):
    """
    Set up main problem/main regularization problem for OA, ECP, Feasibility Pump and ROA methods.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    fp: Bool
        generate the feasibility pump regularization main problem
    regularization_problem: Bool
        generate the ROA regularization main problem
    """
    MindtPy = solve_data.mip.MindtPy_utils

    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in {1, 0}:
            c.deactivate()

    MindtPy.cuts.activate()

    sign_adjust = 1 if solve_data.objective_sense == minimize else - 1
    MindtPy.del_component('mip_obj')
    if regularization_problem and config.single_tree:
        MindtPy.del_component('loa_proj_mip_obj')
        MindtPy.cuts.del_component('obj_reg_estimate')
    if config.add_regularization is not None and config.add_no_good_cuts:
        if regularization_problem:
            MindtPy.cuts.no_good_cuts.activate()
        else:
            MindtPy.cuts.no_good_cuts.deactivate()

    if fp:
        MindtPy.del_component('fp_mip_obj')
        if config.fp_main_norm == 'L1':
            MindtPy.fp_mip_obj = generate_norm1_objective_function(
                solve_data.mip,
                solve_data.working_model,
                discrete_only=config.fp_discrete_only)
        elif config.fp_main_norm == 'L2':
            MindtPy.fp_mip_obj = generate_norm2sq_objective_function(
                solve_data.mip,
                solve_data.working_model,
                discrete_only=config.fp_discrete_only)
        elif config.fp_main_norm == 'L_infinity':
            MindtPy.fp_mip_obj = generate_norm_inf_objective_function(
                solve_data.mip,
                solve_data.working_model,
                discrete_only=config.fp_discrete_only)
    elif regularization_problem:
        if MindtPy.objective_list[0].expr.polynomial_degree() in {1, 0}:
            MindtPy.objective_constr.activate()
        if config.add_regularization == 'level_L1':
            MindtPy.loa_proj_mip_obj = generate_norm1_objective_function(solve_data.mip,
                                                                         solve_data.best_solution_found,
                                                                         discrete_only=False)
        elif config.add_regularization == 'level_L2':
            MindtPy.loa_proj_mip_obj = generate_norm2sq_objective_function(solve_data.mip,
                                                                           solve_data.best_solution_found,
                                                                           discrete_only=False)
        elif config.add_regularization == 'level_L_infinity':
            MindtPy.loa_proj_mip_obj = generate_norm_inf_objective_function(solve_data.mip,
                                                                            solve_data.best_solution_found,
                                                                            discrete_only=False)
        elif config.add_regularization in {'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'}:
            MindtPy.loa_proj_mip_obj = generate_lag_objective_function(solve_data.mip,
                                                                       solve_data.best_solution_found,
                                                                       config,
                                                                       solve_data,
                                                                       discrete_only=False)
        if solve_data.objective_sense == minimize:
            MindtPy.cuts.obj_reg_estimate = Constraint(
                expr=MindtPy.objective_value <= (1 - config.level_coef) * solve_data.UB + config.level_coef * solve_data.LB)
        else:
            MindtPy.cuts.obj_reg_estimate = Constraint(
                expr=MindtPy.objective_value >= (1 - config.level_coef) * solve_data.LB + config.level_coef * solve_data.UB)

    else:
        if config.add_slack:
            MindtPy.del_component('aug_penalty_expr')

            MindtPy.aug_penalty_expr = Expression(
                expr=sign_adjust * config.OA_penalty_factor * sum(
                    v for v in MindtPy.cuts.slack_vars[...]))
        main_objective = MindtPy.objective_list[-1]
        MindtPy.mip_obj = Objective(
            expr=main_objective.expr +
            (MindtPy.aug_penalty_expr if config.add_slack else 0),
            sense=solve_data.objective_sense)

        if config.use_dual_bound:
            # Delete previously added dual bound constraint
            MindtPy.cuts.del_component('dual_bound')
            if solve_data.objective_sense == minimize:
                MindtPy.cuts.dual_bound = Constraint(
                    expr=main_objective.expr +
                    (MindtPy.aug_penalty_expr if config.add_slack else 0) >= solve_data.LB,
                    doc='Objective function expression should improve on the best found dual bound')
            else:
                MindtPy.cuts.dual_bound = Constraint(
                    expr=main_objective.expr +
                    (MindtPy.aug_penalty_expr if config.add_slack else 0) <= solve_data.UB,
                    doc='Objective function expression should improve on the best found dual bound')
