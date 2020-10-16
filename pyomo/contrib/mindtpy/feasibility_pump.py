# -*- coding: utf-8 -*-
from pyomo.core import (Var, Objective, Reals, minimize,
                        RangeSet, Constraint, Block, sqrt, TransformationFactory, ComponentMap, value)
from pyomo.core.base.constraint import ConstraintList
from pyomo.opt import SolverFactory, SolutionStatus
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing, get_main_elapsed_time, copy_var_list_values, is_feasible
from pyomo.contrib.mindtpy.nlp_solve import (solve_subproblem,
                                             handle_subproblem_optimal, handle_subproblem_infeasible,
                                             handle_subproblem_other_termination)
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_nogood_cuts
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy.util import generate_norm2sq_objective_function
from pyomo.contrib.mindtpy.mip_solve import solve_master, handle_master_optimal
from pyomo.util.infeasible import log_infeasible_constraints

from pyomo.contrib.mindtpy.util import generate_norm1_norm_constraint


def feas_pump_converged(solve_data, config, discrete_only=True):
    """Calculates the euclidean norm between the discretes in the mip and nlp models"""
    distance = (max((nlp_var.value - milp_var.value)**2
                    for (nlp_var, milp_var) in
                    zip(solve_data.working_model.MindtPy_utils.variable_list,
                        solve_data.mip.MindtPy_utils.variable_list)
                    if (not discrete_only) or milp_var.is_integer()))
    #
    return distance <= config.fp_projzerotol


def solve_feas_pump_subproblem(solve_data, config):
    """
    Solves the feasibility pump NLP

    This function sets up the 'fp_nlp' by relax integer varibales.
    precomputes dual values, deactivates trivial constraints, and then solves NLP model.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm

    Returns
    -------
    fp_nlp: Pyomo model
        fixed NLP from the model
    results: Pyomo results object
        result from solving the fixed NLP
    """

    fp_nlp = solve_data.working_model.clone()
    MindtPy = fp_nlp.MindtPy_utils
    config.logger.info('FP-NLP %s: Solve feasibility pump NLP subproblem.'
                       % (solve_data.fp_iter,))

    # Set up NLP
    main_objective = next(
        fp_nlp.component_data_objects(Objective, active=True))
    main_objective.deactivate()
    if main_objective.sense == 'minimize':
        fp_nlp.improving_objective_cut = Constraint(
            expr=fp_nlp.MindtPy_utils.objective_value <= solve_data.UB)
    else:
        fp_nlp.improving_objective_cut = Constraint(
            expr=fp_nlp.MindtPy_utils.objective_value >= solve_data.LB)

    # Add norm_constraint, TODO: rename norm_constraint to like improving_distance_cut
    if config.fp_norm_constraint:
        if config.fp_master_norm == 'L1':
            generate_norm1_norm_constraint(
                fp_nlp, solve_data.mip, config, discrete_only=True)
        elif config.fp_master_norm == 'L2':
            fp_nlp.norm_constraint = Constraint(expr=sum((nlp_var - mip_var.value)**2 - config.fp_norm_constraint_coef*(nlp_var.value - mip_var.value)**2
                                                         for nlp_var, mip_var in zip(fp_nlp.MindtPy_utils.variable_list, solve_data.mip.MindtPy_utils.variable_list) if mip_var.is_integer()) <= 0)
        elif config.fp_master_norm == 'L_infinity':
            fp_nlp.norm_constraint = ConstraintList()
            rhs = config.fp_norm_constraint_coef * max(nlp_var.value - mip_var.value for nlp_var, mip_var in zip(
                fp_nlp.MindtPy_utils.variable_list, solve_data.mip.MindtPy_utils.variable_list) if mip_var.is_integer())
            for nlp_var, mip_var in zip(fp_nlp.MindtPy_utils.variable_list, solve_data.mip.MindtPy_utils.variable_list):
                if mip_var.is_integer():
                    fp_nlp.norm_constraint.add(nlp_var - mip_var.value <= rhs)

    MindtPy.feas_pump_nlp_obj = generate_norm2sq_objective_function(
        fp_nlp, solve_data.mip, discrete_only=config.fp_discrete_only)

    MindtPy.MindtPy_linear_cuts.deactivate()
    TransformationFactory('core.relax_integer_vars').apply_to(fp_nlp)
    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
        fp_nlp, tmp=True, ignore_infeasible=True)
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
            fp_nlp, tee=config.nlp_solver_tee, **nlp_args)
    return fp_nlp, results


def handle_feas_pump_subproblem_optimal(fp_nlp, solve_data, config):
    """Copies result to working model, updates bound, adds OA cut, no_good cut
    and increasing objective cut and stores best solution if new one is best
    Also calculates the duals
    """
    copy_var_list_values(
        fp_nlp.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config,
        ignore_integrality=True)
    add_orthogonality_cuts(solve_data, config)

    # if OA-like or feas_pump converged, update Upper bound,
    # add no_good cuts and increasing objective cuts (feas_pump)
    if feas_pump_converged(solve_data, config, discrete_only=config.fp_discrete_only):
        copy_var_list_values(solve_data.mip.MindtPy_utils.variable_list,
                             solve_data.working_model.MindtPy_utils.variable_list,
                             config)
        fixed_nlp, fixed_nlp_results = solve_subproblem(
            solve_data, config)
        main_objective = next(
            fixed_nlp.component_data_objects(Objective, active=True))
        if fixed_nlp_results.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            handle_subproblem_optimal(
                fixed_nlp, solve_data, config, feas_pump=True)
        else:
            config.logger.error("Feasibility pump fixed nlp is infeasible, something might be wrong. "
                                "There might be a problem with the precisions - the feasibility pump seems to have converged")

    if solve_data.solution_improved:
        solve_data.best_solution_found = solve_data.working_model.clone()
        # log_infeasible_constraints(solve_data.working_model)


def feas_pump_loop(solve_data, config):
    """
    Feasibility pump loop 

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
        feas_master, feas_master_results = solve_master(
            solve_data, config, feas_pump=True)
        if feas_master_results.solver.termination_condition is tc.optimal:
            config.logger.info(
                'FP-MIP %s: Distance-OBJ: %s'
                % (solve_data.fp_iter, value(solve_data.mip.MindtPy_utils.feas_pump_mip_obj)))
        elif feas_master_results.solver.termination_condition is tc.maxTimeLimit:
            config.logger.warning('FP-MIP reaches max TimeLimit')
        elif feas_master_results.solver.termination_condition is tc.infeasible:
            config.logger.warning('FP-MIP infeasible')
            # TODO: needs to be checked here.
            nogood_cuts = solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.nogood_cuts
            if nogood_cuts.__len__() > 0:
                nogood_cuts[nogood_cuts.__len__()].deactivate()
            break
        elif feas_master_results.solver.termination_condition is tc.unbounded:
            config.logger.warning('FP-MIP unbounded')
            break
        elif (feas_master_results.solver.termination_condition is tc.other and
              feas_master_results.solution.status is SolutionStatus.feasible):
            config.logger.warning('MILP solver reported feasible solution of FP-MIP, '
                                  'but not guaranteed to be optimal.')
        else:
            config.logger.warning('Unexpected result of FP-MIP')
            break

            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
        fp_nlp, fp_nlp_result = solve_feas_pump_subproblem(
            solve_data, config)

        if fp_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            config.logger.info('FP-NLP %s: Distance-OBJ: %s'
                               % (solve_data.fp_iter, value(fp_nlp.MindtPy_utils.feas_pump_nlp_obj)))
            handle_feas_pump_subproblem_optimal(fp_nlp, solve_data, config)
        elif fp_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
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
    # solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.fp_orthogonality_cuts.deactivate()
    # deactivate the improving_objective_cut
    if solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.find_component('improving_objective_cut') is not None:
        solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.improving_objective_cut.deactivate()
    if not config.fp_transfercuts:
        for c in solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.oa_cuts:
            c.deactivate()
        for c in solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.nogood_cuts:
            c.deactivate()
    if config.fp_projcuts:
        solve_data.working_model.MindtPy_utils.MindtPy_linear_cuts.del_component(
            'fp_orthogonality_cuts')


def add_orthogonality_cuts(solve_data, config):
    """
    Add orthogonality cuts

    This function adds orthogonality cuts to avoid cycling when the independence constraint qualification is not satisfied.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    m = solve_data.mip
    mip_MindtPy = solve_data.mip.MindtPy_utils
    nlp_MindtPy = solve_data.working_model.MindtPy_utils
    mip_integer_vars = [v for v in mip_MindtPy.variable_list if v.is_integer()]
    nlp_integer_vars = [v for v in nlp_MindtPy.variable_list if v.is_integer()]
    orthogonality_cut = sum((nlp_v.value-mip_v.value)*(mip_v-nlp_v.value)
                            for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)) >= 0
    solve_data.mip.MindtPy_utils.MindtPy_linear_cuts.fp_orthogonality_cuts.add(
        orthogonality_cut)
    if config.fp_projcuts:
        orthogonality_cut = sum((nlp_v.value-mip_v.value)*(nlp_v-nlp_v.value)
                                for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)) >= 0
        solve_data.working_model.MindtPy_utils.MindtPy_linear_cuts.fp_orthogonality_cuts.add(
            orthogonality_cut)
