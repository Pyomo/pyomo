from pyomo.core import (Var, Objective, Reals, minimize,
                        RangeSet, Constraint, Block, sqrt, TransformationFactory, ComponentMap, value)
from pyomo.opt import SolverFactory, SolutionStatus
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing, get_main_elapsed_time, copy_var_list_values, is_feasible
from pyomo.contrib.mindtpy.nlp_solve import (solve_NLP_subproblem,
                                             handle_NLP_subproblem_optimal, handle_NLP_subproblem_infeasible,
                                             handle_NLP_subproblem_other_termination)
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_nogood_cuts
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy.util import generate_Norm2sq_objective_function
from pyomo.contrib.mindtpy.mip_solve import solve_MIP_master, handle_master_mip_optimal


def feas_pump_converged(solve_data, config, discrete_only=True):
    """Calculates the euclidean norm between the discretes in the mip and nlp models"""
    distance = (sum((nlp_var.value - milp_var.value)**2
                    for (nlp_var, milp_var) in
                    zip(solve_data.working_model.MindtPy_utils.variable_list,
                        solve_data.mip.MindtPy_utils.variable_list)
                    if (not discrete_only) or milp_var.is_binary()))

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
    config.logger.info('FP-NLP %s: Solve feasibility pump NLP subproblem.'
                       % (solve_data.fp_iter,))

    # Set up NLP
    TransformationFactory('core.relax_integer_vars').apply_to(sub_nlp)
    main_objective = next(
        sub_nlp.component_data_objects(Objective, active=True))
    main_objective.deactivate()
    # TODO: need to comfirm with David, whether to add increasing_objective_cut for FP-NLP
    # sub_nlp may don't have MindtPy_utils.objective_value
    # if main_objective.sense == 'minimize':
    #     sub_nlp.improving_objective_cut = Constraint(
    #         expr=sub_nlp.MindtPy_utils.objective_value <= solve_data.UB)
    # else:
    #     sub_nlp.improving_objective_cut = Constraint(
    #         expr=sub_nlp.MindtPy_utils.objective_value >= solve_data.LB)
    MindtPy.feas_pump_nlp_obj = generate_Norm2sq_objective_function(
        sub_nlp, solve_data.mip, discrete_only=True)

    MindtPy.MindtPy_linear_cuts.deactivate()
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

    # if OA-like or feas_pump converged, update Upper bound,
    # add no_good cuts and increasing objective cuts (feas_pump)
    if feas_pump_converged(solve_data, config):
        copy_var_list_values(solve_data.mip.MindtPy_utils.variable_list,
                             solve_data.working_model.MindtPy_utils.variable_list,
                             config)
        fixed_nlp, fixed_nlp_results = solve_NLP_subproblem(
            solve_data, config)
        main_objective = next(
            fixed_nlp.component_data_objects(Objective, active=True))
        # assert fixed_nlp_results.solver.termination_condition is tc.optimal, 'Feasibility pump fixed_nlp subproblem not optimal'
        if fixed_nlp_results.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            handle_NLP_subproblem_optimal(
                fixed_nlp, solve_data, config, feas_pump=True)
        else:
            config.logger.error("Feasibility pump fixed nlp is infeasible, something might be wrong. "
                                "There might be a problem with the precisions - the feasibility pump seems to have converged")

    if solve_data.solution_improved:
        solve_data.best_solution_found = solve_data.working_model.clone()
        assert is_feasible(solve_data.best_solution_found, config), \
            "Best found solution infeasible! There might be a problem with the precisions - the feasibility pump seems to have converged (error**2 <= integer_tolerance). " \
            "But the `is_feasible` check (error <= constraint_tolerance) doesn't work out"


def feas_pump_loop(solve_data, config):
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
            config.logger.info(
                'FP-MIP %s: Distance-OBJ: %s'
                % (solve_data.fp_iter, value(solve_data.mip.MindtPy_utils.feas_pump_mip_obj)))
        elif feas_mip_results.solver.termination_condition is tc.maxTimeLimit:
            config.logger.warning('FP-MIP reaches max TimeLimit')
        elif feas_mip_results.solver.termination_condition is tc.infeasible:
            config.logger.warning('FP-MIP infeasible')
            break
        elif feas_mip_results.solver.termination_condition is tc.unbounded:
            config.logger.warning('FP-MIP unbounded')
            break
        elif (feas_mip_results.solver.termination_condition is tc.other and
              feas_mip_results.solution.status is SolutionStatus.feasible):
            config.logger.warning('MILP solver reported feasible solution of FP-MIP, '
                                  'but not guaranteed to be optimal.')
        else:
            config.logger.warning('Unexpected result of FP-MIP')
            break

            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
        fp_nlp, fp_nlp_result = solve_feas_pump_NLP_subproblem(
            solve_data, config)

        if fp_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            handle_feas_pump_NLP_subproblem_optimal(fp_nlp, solve_data, config)
            config.logger.info(
                'FP-NLP %s: Distance-OBJ: %s'
                % (solve_data.fp_iter, value(fp_nlp.MindtPy_utils.feas_pump_nlp_obj)))
        elif fp_nlp_result.solver.termination_condition is tc.infeasible:
            config.logger.error("Feasibility pump NLP subproblem infeasible")
        elif termination_condition is tc.maxIterations:
            config.logger.info(
                'Feasibility pump NLP subproblem failed to converge within iteration limit.')
        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(termination_condition))
        # Call the NLP post-solve callback
        config.call_after_subproblem_solve(fp_nlp, solve_data)
        solve_data.fp_iter += 1
