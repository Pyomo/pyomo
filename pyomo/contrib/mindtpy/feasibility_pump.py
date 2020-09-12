"""Generates objective functions like L1, L2 and Linf distance"""

from pyomo.core import (Var, Objective, Reals, minimize,
                        RangeSet, Constraint, Block, sqrt, TransformationFactory, ComponentMap, value)
from pyomo.opt import SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing, get_main_elapsed_time, copy_var_list_values, is_feasible
from pyomo.contrib.mindtpy.nlp_solve import (solve_NLP_subproblem,
                                             handle_NLP_subproblem_optimal, handle_NLP_subproblem_infeasible,
                                             handle_NLP_subproblem_other_termination)
from pyomo.contrib.mindtpy.mip_solve import solve_MIP_master
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_nogood_cuts
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy.util import generate_L2_objective_function

def feas_pump_converged(solve_data, config):
    """Calculates the euclidean norm between the discretes in the mip and nlp models"""
    distance = (sum((nlp_var.value - milp_var.value)**2
                    for (nlp_var, milp_var) in
                    zip(solve_data.working_model.MindtPy_utils.variable_list,
                        solve_data.mip.MindtPy_utils.variable_list)
                    if milp_var.is_binary()))

    return distance <= config.integer_tolerance


def solve_feas_pump_NLP_subproblem(solve_data, config):
    """
    Solves the fixed NLP (with fixed binaries)

    This function sets up the 'sub_nlp' by fixing binaries, sets continuous variables to their intial var values,
    precomputes dual values, deactivates trivial constraints, and then solves NLP model.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    Returns
    -------
    sub_nlp: Pyomo model
        fixed NLP from the model
    results: Pyomo results object
        result from solving the fixed NLP
    """

    sub_nlp = solve_data.working_model.clone()
    MindtPy = sub_nlp.MindtPy_utils
    solve_data.nlp_iter += 1
    config.logger.info('NLP %s: Solve subproblem for fixed binaries.'
                       % (solve_data.nlp_iter,))

    # Set up NLP
    TransformationFactory('core.relax_integrality').apply_to(sub_nlp)
    main_objective = next(
        sub_nlp.component_data_objects(Objective, active=True))
    main_objective.deactivate()
    if main_objective.sense == 'minimize':
        sub_nlp.increasing_objective_cut = Constraint(
            expr=sub_nlp.MindtPy_utils.objective_value
            <= solve_data.UB - config.feas_pump_delta*min(1e-4, abs(solve_data.UB)))
    else:
        sub_nlp.increasing_objective_cut = Constraint(
            expr=sub_nlp.MindtPy_utils.objective_value
            >= solve_data.LB + config.feas_pump_delta*min(1e-4, abs(solve_data.LB)))
    MindtPy.feas_pump_nlp_obj = generate_L2_objective_function(
        sub_nlp,
        solve_data.mip,
        discretes_only=True
    )

    MindtPy.MindtPy_linear_cuts.deactivate()
    sub_nlp.tmp_duals = ComponentMap()
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
    for c in sub_nlp.component_data_objects(ctype=Constraint, active=True,
                                            descend_into=True):
        # We prefer to include the upper bound as the right hand side since we are
        # considering c by default a (hopefully) convex function, which would make
        # c >= lb a nonconvex inequality which we wouldn't like to add linearizations
        # if we don't have to
        rhs = c.upper if c.has_ub() else c.lower
        c_geq = -1 if c.has_ub() else 1
        # c_leq = 1 if c.has_ub else -1
        try:
            sub_nlp.tmp_duals[c] = c_geq * max(
                0, c_geq*(rhs - value(c.body)))
        except (ValueError, OverflowError) as error:
            sub_nlp.tmp_duals[c] = None
            evaluation_error = True
    if evaluation_error:
        for nlp_var, orig_val in zip(
                MindtPy.variable_list,
                solve_data.initial_var_values):
            if not nlp_var.fixed and not nlp_var.is_binary():
                nlp_var.value = orig_val
        # sub_nlp.tmp_duals[c] = c_leq * max(
        #     0, c_leq*(value(c.body) - rhs))
        # TODO: change logic to c_leq based on benchmarking

    TransformationFactory('contrib.deactivate_trivial_constraints')\
        .apply_to(sub_nlp, tmp=True, ignore_infeasible=True)
    # Solve the NLP
    nlpopt = SolverFactory(config.nlp_solver)
    nlp_args = dict(config.nlp_solver_args)
    elapsed = get_main_elapsed_time(solve_data.timing)
    remaining = int(max(config.time_limit - elapsed, 1))
    if config.nlp_solver == 'gams':
        nlp_args['add_options'] = nlp_args.get('add_options', [])
        nlp_args['add_options'].append('option reslim=%s;' % remaining)
    with SuppressInfeasibleWarning():
        results = nlpopt.solve(
            sub_nlp, tee=config.solver_tee, **nlp_args)
    return sub_nlp, results


def handle_feas_pump_NLP_subproblem_optimal(sub_nlp, solve_data, config):
    """Copies result to working model, updates bound, adds OA cut, no_good cut
    and increasing objective cut and stores best solution if new one is best
    Also calculates the duals
    """
    copy_var_list_values(
        sub_nlp.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config,
        ignore_integrality=config.strategy == 'feas_pump')

    for c in sub_nlp.tmp_duals:
        if sub_nlp.dual.get(c, None) is None:
            sub_nlp.dual[c] = sub_nlp.tmp_duals[c]
    dual_values = list(sub_nlp.dual[c]
                       for c in sub_nlp.MindtPy_utils.constraint_list)

    main_objective = next(
        solve_data.working_model.component_data_objects(
            Objective,
            active=True))  # this is different to original objective for feasibility pump

    # if OA-like or feas_pump converged, update Upper bound,
    # add no_good cuts and increasing objective cuts (feas_pump)
    if feas_pump_converged(solve_data, config):
        copy_var_list_values(solve_data.mip.MindtPy_utils.variable_list,
                             solve_data.working_model.MindtPy_utils.variable_list,
                             config)
        fixed_nlp, fixed_nlp_results = solve_NLP_subproblem(
            solve_data, config)
        assert fixed_nlp_results.solver.termination_condition is tc.optimal, 'Feasibility pump fixed_nlp subproblem not optimal'
        copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list,
                             solve_data.working_model.MindtPy_utils.variable_list,
                             config)
        if main_objective.sense == minimize:
            solve_data.UB = min(main_objective.expr(), solve_data.UB)
            solve_data.solution_improved = solve_data.UB < solve_data.UB_progress[-1]
            solve_data.UB_progress.append(solve_data.UB)

            if solve_data.solution_improved and config.strategy == 'feas_pump':
                solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.\
                    increasing_objective_cut.set_value(
                        expr=solve_data.mip.MindtPy_utils.objective_value
                        <= solve_data.UB - config.feas_pump_delta*min(1e-4, abs(solve_data.UB)))
        else:
            solve_data.LB = max(main_objective.expr(), solve_data.LB)
            solve_data.solution_improved = solve_data.LB > solve_data.LB_progress[-1]
            solve_data.LB_progress.append(solve_data.LB)

            if solve_data.solution_improved and config.strategy == 'feas_pump':
                solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.\
                    increasing_objective_cut.set_value(
                        expr=solve_data.mip.MindtPy_utils.objective_value
                        >= solve_data.LB + config.feas_pump_delta*min(1e-4, abs(solve_data.LB)))

        if config.add_no_good_cuts:
            config.logger.info('Creating no-good cut')
            add_nogood_cuts(solve_data.mip, config)
    else:
        solve_data.solution_improved = False

    config.logger.info(
        'NLP {}: OBJ: {}  LB: {}  UB: {}'
        .format(solve_data.nlp_iter,
                main_objective.expr(),
                solve_data.LB, solve_data.UB))

    if solve_data.solution_improved:
        solve_data.best_solution_found = solve_data.working_model.clone()
        assert is_feasible(solve_data.best_solution_found, config), \
            "Best found solution infeasible! There might be a problem with the precisions - the feaspump seems to have converged (error**2 <= integer_tolerance). " \
            "But the `is_feasible` check (error <= constraint_tolerance) doesn't work out"

    # Always add the oa cut
    copy_var_list_values(sub_nlp.MindtPy_utils.variable_list,
                         solve_data.mip.MindtPy_utils.variable_list,
                         config, ignore_integrality=True)
    add_oa_cuts(solve_data.mip, dual_values, solve_data, config)

    config.call_after_subproblem_feasible(sub_nlp, solve_data)


def feasibility_pump_loop(solve_data, config):
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
    while solve_data.fp_iter < config.fp_iteration_limit:

        config.logger.info(
            '---Feasibility Pump Iteration %s---'
            % solve_data.fp_iter)

        solve_data.mip_subiter = 0
        # solve MILP master problem
        feas_mip, feas_mip_results = solve_MIP_master(
            solve_data, config, feas_pump=True)
        if feas_mip_results.solver.termination_condition is tc.optimal:
            handle_master_mip_optimal(feas_mip, solve_data, config)
        elif feas_mip_results.solver.termination_condition is tc.infeasible:
            # This basically means the incumbent is the optimal solution
            if solve_data.best_solution_found is not None:
                config.logger.info(
                    'Problem became infeasible. This means the feasibility pump has converged.')
                solve_data.results.solver.termination_condition = tc.optimal
            else:
                config.logger.info('No feasible solution has been found')
                solve_data.results.solver.termination_condition = tc.infeasible
            break
        elif feas_mip_results.solver.termination_condition is tc.maxIterations:
            config.logger.error('No feasible solution has been found')
            solve_data.results.solver.termination_condition = tc.maxIterations
            break

        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        fixed_nlp, fixed_nlp_result = solve_feas_pump_NLP_subproblem(
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
        solve_data.fp_iter += 1
