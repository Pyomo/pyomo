#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Solution of NLP subproblems."""
from __future__ import division
import logging
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mindtpy.cut_generation import (add_oa_cuts,
                                                  add_no_good_cuts, add_affine_cuts)
from pyomo.contrib.mindtpy.util import add_feas_slacks, set_solver_options
from pyomo.contrib.gdpopt.util import copy_var_list_values, get_main_elapsed_time, time_code
from pyomo.core import (Constraint, Objective,
                        TransformationFactory, minimize, value)
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory, SolverResults, SolverStatus
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning

logger = logging.getLogger('pyomo.contrib.mindtpy')


def solve_subproblem(solve_data, config):
    """
    Solves the Fixed-NLP (with fixed integers)

    This function sets up the 'fixed_nlp' by fixing binaries, sets continuous variables to their intial var values,
    precomputes dual values, deactivates trivial constraints, and then solves NLP model.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    Returns
    -------
    fixed_nlp: Pyomo model
        integer-variable-fixed NLP model
    results: Pyomo results object
        result from solving the Fixed-NLP
    """

    fixed_nlp = solve_data.working_model.clone()
    MindtPy = fixed_nlp.MindtPy_utils
    solve_data.nlp_iter += 1
    config.logger.info('Fixed-NLP %s: Solve subproblem for fixed integers.'
                       % (solve_data.nlp_iter,))

    # Set up NLP
    TransformationFactory('core.fix_integer_vars').apply_to(fixed_nlp)

    MindtPy.cuts.deactivate()
    if config.calculate_dual:
        fixed_nlp.tmp_duals = ComponentMap()
        # tmp_duals are the value of the dual variables stored before using deactivate trivial contraints
        # The values of the duals are computed as follows: (Complementary Slackness)
        #
        # | constraint | c_geq | status at x1 | tmp_dual (violation) |
        # |------------|-------|--------------|----------------------|
        # | g(x) <= b  | -1    | g(x1) <= b   | 0                    |
        # | g(x) <= b  | -1    | g(x1) > b    | g(x1) - b            |
        # | g(x) >= b  | +1    | g(x1) >= b   | 0                    |
        # | g(x) >= b  | +1    | g(x1) < b    | b - g(x1)            |
        evaluation_error = False
        for c in fixed_nlp.MindtPy_utils.constraint_list:
            # We prefer to include the upper bound as the right hand side since we are
            # considering c by default a (hopefully) convex function, which would make
            # c >= lb a nonconvex inequality which we wouldn't like to add linearizations
            # if we don't have to
            rhs = value(c.upper) if c.has_ub() else value(c.lower)
            c_geq = -1 if c.has_ub() else 1
            # c_leq = 1 if c.has_ub else -1
            try:
                fixed_nlp.tmp_duals[c] = c_geq * max(
                    0, c_geq*(rhs - value(c.body)))
            except (ValueError, OverflowError) as error:
                fixed_nlp.tmp_duals[c] = None
                evaluation_error = True
        if evaluation_error:
            for nlp_var, orig_val in zip(
                    MindtPy.variable_list,
                    solve_data.initial_var_values):
                if not nlp_var.fixed and not nlp_var.is_binary():
                    nlp_var.value = orig_val
            # fixed_nlp.tmp_duals[c] = c_leq * max(
            #     0, c_leq*(value(c.body) - rhs))
            # TODO: change logic to c_leq based on benchmarking
    try:
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
            fixed_nlp, tmp=True, ignore_infeasible=False, tolerance=config.constraint_tolerance)
    except ValueError:
        config.logger.warning(
            'infeasibility detected in deactivate_trivial_constraints')
        results = SolverResults()
        results.solver.termination_condition = tc.infeasible
        return fixed_nlp, results
    # Solve the NLP
    nlpopt = SolverFactory(config.nlp_solver)
    nlp_args = dict(config.nlp_solver_args)
    set_solver_options(nlpopt, solve_data, config, solver_type='nlp')
    with SuppressInfeasibleWarning():
        with time_code(solve_data.timing, 'fixed subproblem'):
            results = nlpopt.solve(
                fixed_nlp, tee=config.nlp_solver_tee, **nlp_args)
    return fixed_nlp, results


def handle_nlp_subproblem_tc(fixed_nlp, result, solve_data, config, cb_opt=None):
    '''
    This function handles different terminaton conditions of the fixed-NLP subproblem.

    Parameters
    ----------
    fixed_nlp: Pyomo model
        integer-variable-fixed NLP model
    results: Pyomo results object
        result from solving the Fixed-NLP
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    cb_opt: SolverFactory
            the gurobi_persistent solver
    '''
    if result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
        handle_subproblem_optimal(fixed_nlp, solve_data, config, cb_opt)
    elif result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
        handle_subproblem_infeasible(fixed_nlp, solve_data, config, cb_opt)
    elif result.solver.termination_condition is tc.maxTimeLimit:
        config.logger.info(
            'NLP subproblem failed to converge within the time limit.')
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
        solve_data.should_terminate = True
    elif result.solver.termination_condition is tc.maxEvaluations:
        config.logger.info(
            'NLP subproblem failed due to maxEvaluations.')
        solve_data.results.solver.termination_condition = tc.maxEvaluations
        solve_data.should_terminate = True
    else:
        handle_subproblem_other_termination(fixed_nlp, result.solver.termination_condition,
                                            solve_data, config)


# The next few functions deal with handling the solution we get from the above NLP solver function


def handle_subproblem_optimal(fixed_nlp, solve_data, config, cb_opt=None, fp=False):
    """
    This function copies the result of the NLP solver function ('solve_subproblem') to the working model, updates
    the bounds, adds OA and no-good cuts, and then stores the new solution if it is the new best solution. This
    function handles the result of the latest iteration of solving the NLP subproblem given an optimal solution.

    Parameters
    ----------
    fixed_nlp: Pyomo model
        integer-variable-fixed NLP model
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    cb_opt: SolverFactory
        the gurobi_persistent solver
    fp: bool, optional
        this parameter acts as a Boolean flag that signals whether it is in the loop of feasibility pump
    """
    copy_var_list_values(
        fixed_nlp.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)
    if config.calculate_dual:
        for c in fixed_nlp.tmp_duals:
            if fixed_nlp.dual.get(c, None) is None:
                fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
        dual_values = list(fixed_nlp.dual[c]
                           for c in fixed_nlp.MindtPy_utils.constraint_list)
    else:
        dual_values = None
    main_objective = fixed_nlp.MindtPy_utils.objective_list[-1]
    if solve_data.objective_sense == minimize:
        solve_data.UB = min(value(main_objective.expr), solve_data.UB)
        solve_data.solution_improved = solve_data.UB < solve_data.UB_progress[-1]
        solve_data.UB_progress.append(solve_data.UB)
    else:
        solve_data.LB = max(value(main_objective.expr), solve_data.LB)
        solve_data.solution_improved = solve_data.LB > solve_data.LB_progress[-1]
        solve_data.LB_progress.append(solve_data.LB)
    config.logger.info(
        'Fixed-NLP {}: OBJ: {}  LB: {}  UB: {}  TIME: {}s'
        .format(solve_data.nlp_iter if not fp else solve_data.fp_iter, value(main_objective.expr),
                solve_data.LB, solve_data.UB, round(get_main_elapsed_time(solve_data.timing), 2)))

    if solve_data.solution_improved:
        solve_data.best_solution_found = fixed_nlp.clone()
        solve_data.best_solution_found_time = get_main_elapsed_time(
            solve_data.timing)
        if config.strategy == 'GOA':
            if solve_data.objective_sense == minimize:
                solve_data.num_no_good_cuts_added.update(
                    {solve_data.UB: len(solve_data.mip.MindtPy_utils.cuts.no_good_cuts)})
            else:
                solve_data.num_no_good_cuts_added.update(
                    {solve_data.LB: len(solve_data.mip.MindtPy_utils.cuts.no_good_cuts)})

        # add obj increasing constraint for fp
        if fp:
            solve_data.mip.MindtPy_utils.cuts.del_component(
                'improving_objective_cut')
            if solve_data.objective_sense == minimize:
                solve_data.mip.MindtPy_utils.cuts.improving_objective_cut = Constraint(expr=solve_data.mip.MindtPy_utils.objective_value
                                                                                       <= solve_data.UB - config.fp_cutoffdecr*max(1, abs(solve_data.UB)))
            else:
                solve_data.mip.MindtPy_utils.cuts.improving_objective_cut = Constraint(expr=solve_data.mip.MindtPy_utils.objective_value
                                                                                       >= solve_data.LB + config.fp_cutoffdecr*max(1, abs(solve_data.UB)))

    # Add the linear cut
    if config.strategy == 'OA' or fp:
        copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                             solve_data.mip.MindtPy_utils.variable_list,
                             config)
        add_oa_cuts(solve_data.mip, dual_values, solve_data, config, cb_opt)
    elif config.strategy == 'GOA':
        copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                             solve_data.mip.MindtPy_utils.variable_list,
                             config)
        add_affine_cuts(solve_data, config)
    # elif config.strategy == 'PSC':
    #     # !!THIS SEEMS LIKE A BUG!! - mrmundt #
    #     add_psc_cut(solve_data, config)
    # elif config.strategy == 'GBD':
    #     # !!THIS SEEMS LIKE A BUG!! - mrmundt #
    #     add_gbd_cut(solve_data, config)

    var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
    if config.add_no_good_cuts:
        add_no_good_cuts(var_values, solve_data, config, feasible=True)

    config.call_after_subproblem_feasible(fixed_nlp, solve_data)


def handle_subproblem_infeasible(fixed_nlp, solve_data, config, cb_opt=None):
    """
    Solves feasibility problem and adds cut according to the specified strategy

    This function handles the result of the latest iteration of solving the NLP subproblem given an infeasible
    solution and copies the solution of the feasibility problem to the working model.

    Parameters
    ----------
    fixed_nlp: Pyomo model
        integer-variable-fixed NLP model
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    cb_opt: SolverFactory
            the gurobi_persistent solver
    """
    # TODO try something else? Reinitialize with different initial
    # value?
    config.logger.info('NLP subproblem was locally infeasible.')
    solve_data.nlp_infeasible_counter += 1
    if config.calculate_dual:
        for c in fixed_nlp.MindtPy_utils.constraint_list:
            rhs = value(c.upper) if c. has_ub() else value(c.lower)
            c_geq = -1 if c.has_ub() else 1
            fixed_nlp.dual[c] = (c_geq
                                 * max(0, c_geq * (rhs - value(c.body))))
        dual_values = list(fixed_nlp.dual[c]
                           for c in fixed_nlp.MindtPy_utils.constraint_list)
    else:
        dual_values = None

    # if config.strategy == 'PSC' or config.strategy == 'GBD':
    #     for var in fixed_nlp.component_data_objects(ctype=Var, descend_into=True):
    #         fixed_nlp.ipopt_zL_out[var] = 0
    #         fixed_nlp.ipopt_zU_out[var] = 0
    #         if var.has_ub() and abs(var.ub - value(var)) < config.bound_tolerance:
    #             fixed_nlp.ipopt_zL_out[var] = 1
    #         elif var.has_lb() and abs(value(var) - var.lb) < config.bound_tolerance:
    #             fixed_nlp.ipopt_zU_out[var] = -1

    if config.strategy in {'OA', 'GOA'}:
        config.logger.info('Solving feasibility problem')
        feas_subproblem, feas_subproblem_results = solve_feasibility_subproblem(
            solve_data, config)
        # TODO: do we really need this?
        if solve_data.should_terminate:
            return
        copy_var_list_values(feas_subproblem.MindtPy_utils.variable_list,
                             solve_data.mip.MindtPy_utils.variable_list,
                             config)
        if config.strategy == 'OA':
            add_oa_cuts(solve_data.mip, dual_values,
                        solve_data, config, cb_opt)
        elif config.strategy == 'GOA':
            add_affine_cuts(solve_data, config)
    # Add a no-good cut to exclude this discrete option
    var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
    if config.add_no_good_cuts:
        # excludes current discrete option
        add_no_good_cuts(var_values, solve_data, config)


def handle_subproblem_other_termination(fixed_nlp, termination_condition,
                                        solve_data, config):
    """
    Handles the result of the latest iteration of solving the NLP subproblem given a solution that is neither optimal
    nor infeasible.

    Parameters
    ----------
    termination_condition: Pyomo TerminationCondition
        the termination condition of the NLP subproblem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    if termination_condition is tc.maxIterations:
        # TODO try something else? Reinitialize with different initial value?
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
        var_values = list(
            v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        if config.add_no_good_cuts:
            # excludes current discrete option
            add_no_good_cuts(var_values, solve_data, config)

    else:
        raise ValueError(
            'MindtPy unable to handle NLP subproblem termination '
            'condition of {}'.format(termination_condition))


def solve_feasibility_subproblem(solve_data, config):
    """
    Solves a feasibility NLP if the fixed_nlp problem is infeasible

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    Returns
    -------
    feas_subproblem: Pyomo model
        feasibility NLP from the model
    feas_soln: Pyomo results object
        result from solving the feasibility NLP
    """
    feas_subproblem = solve_data.working_model.clone()
    add_feas_slacks(feas_subproblem, config)

    MindtPy = feas_subproblem.MindtPy_utils
    if MindtPy.find_component('objective_value') is not None:
        MindtPy.objective_value.value = 0

    next(feas_subproblem.component_data_objects(
        Objective, active=True)).deactivate()
    for constr in feas_subproblem.MindtPy_utils.nonlinear_constraint_list:
        constr.deactivate()

    MindtPy.feas_opt.activate()
    if config.feasibility_norm == 'L1':
        MindtPy.feas_obj = Objective(
            expr=sum(s for s in MindtPy.feas_opt.slack_var[...]),
            sense=minimize)
    elif config.feasibility_norm == 'L2':
        MindtPy.feas_obj = Objective(
            expr=sum(s*s for s in MindtPy.feas_opt.slack_var[...]),
            sense=minimize)
    else:
        MindtPy.feas_obj = Objective(
            expr=MindtPy.feas_opt.slack_var,
            sense=minimize)
    TransformationFactory('core.fix_integer_vars').apply_to(feas_subproblem)
    nlpopt = SolverFactory(config.nlp_solver)
    nlp_args = dict(config.nlp_solver_args)
    set_solver_options(nlpopt, solve_data, config, solver_type='nlp')
    with SuppressInfeasibleWarning():
        try:
            with time_code(solve_data.timing, 'feasibility subproblem'):
                feas_soln = nlpopt.solve(
                    feas_subproblem, tee=config.nlp_solver_tee, **nlp_args)
        except (ValueError, OverflowError) as error:
            for nlp_var, orig_val in zip(
                    MindtPy.variable_list,
                    solve_data.initial_var_values):
                if not nlp_var.fixed and not nlp_var.is_binary():
                    nlp_var.value = orig_val
            with time_code(solve_data.timing, 'feasibility subproblem'):
                feas_soln = nlpopt.solve(
                    feas_subproblem, tee=config.nlp_solver_tee, **nlp_args)
    handle_feasibility_subproblem_tc(
        feas_soln.solver.termination_condition, MindtPy, solve_data, config)
    return feas_subproblem, feas_soln


def handle_feasibility_subproblem_tc(subprob_terminate_cond, MindtPy, solve_data, config):
    if subprob_terminate_cond in {tc.optimal, tc.locallyOptimal, tc.feasible}:
        copy_var_list_values(
            MindtPy.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
        if value(MindtPy.feas_obj.expr) <= config.zero_tolerance:
            config.logger.warning('The objective value %.4E of feasibility problem is less than zero_tolerance. '
                                  'This indicates that the nlp subproblem is feasible, although it is found infeasible in the previous step. '
                                  'Check the nlp solver output' % value(MindtPy.feas_obj.expr))
    elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
        config.logger.error('Feasibility subproblem infeasible. '
                            'This should never happen.')
        solve_data.should_terminate = True
        solve_data.results.solver.status = SolverStatus.error
    elif subprob_terminate_cond is tc.maxIterations:
        config.logger.error('Subsolver reached its maximum number of iterations without converging, '
                            'consider increasing the iterations limit of the subsolver or reviewing your formulation.')
        solve_data.should_terminate = True
        solve_data.results.solver.status = SolverStatus.error
    else:
        config.logger.error('MindtPy unable to handle feasibility subproblem termination condition '
                            'of {}'.format(subprob_terminate_cond))
        solve_data.should_terminate = True
        solve_data.results.solver.status = SolverStatus.error
