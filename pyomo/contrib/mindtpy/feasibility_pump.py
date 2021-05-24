#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import (minimize, Constraint, TransformationFactory, value)
from pyomo.core.base.constraint import ConstraintList
from pyomo.opt import SolverFactory, SolutionStatus, SolverResults, SolverStatus
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, copy_var_list_values, time_code
from pyomo.contrib.mindtpy.nlp_solve import solve_subproblem, handle_subproblem_optimal
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy.util import generate_norm2sq_objective_function, set_solver_options
from pyomo.contrib.mindtpy.mip_solve import solve_main
from pyomo.contrib.mindtpy.util import generate_norm1_norm_constraint


def fp_converged(solve_data, config, discrete_only=True):
    """Calculates the euclidean norm between the discretes in the mip and nlp models"""
    distance = (max((nlp_var.value - milp_var.value)**2
                    for (nlp_var, milp_var) in
                    zip(solve_data.working_model.MindtPy_utils.variable_list,
                        solve_data.mip.MindtPy_utils.variable_list)
                    if (not discrete_only) or milp_var.is_integer()))
    return distance <= config.fp_projzerotol


def solve_fp_subproblem(solve_data, config):
    """
    Solves the feasibility pump NLP

    This function sets up the 'fp_nlp' by relax integer variables.
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
        Fixed-NLP from the model
    results: Pyomo results object
        result from solving the Fixed-NLP
    """

    fp_nlp = solve_data.working_model.clone()
    MindtPy = fp_nlp.MindtPy_utils
    config.logger.info('FP-NLP %s: Solve feasibility pump NLP subproblem.'
                       % (solve_data.fp_iter,))

    # Set up NLP
    fp_nlp.MindtPy_utils.objective_list[-1].deactivate()
    if solve_data.objective_sense == minimize:
        fp_nlp.improving_objective_cut = Constraint(
            expr=fp_nlp.MindtPy_utils.objective_value <= solve_data.UB)
    else:
        fp_nlp.improving_objective_cut = Constraint(
            expr=fp_nlp.MindtPy_utils.objective_value >= solve_data.LB)

    # Add norm_constraint, which guarantees the monotonicity of the norm objective value sequence of all iterations
    # Ref: Paper 'A storm of feasibility pumps for nonconvex MINLP'
    # the norm type is consistant with the norm obj of the FP-main problem.
    if config.fp_norm_constraint:
        if config.fp_main_norm == 'L1':
            # TODO: check if we can access the block defined in FP-main problem
            generate_norm1_norm_constraint(
                fp_nlp, solve_data.mip, config, discrete_only=True)
        elif config.fp_main_norm == 'L2':
            fp_nlp.norm_constraint = Constraint(expr=sum((nlp_var - mip_var.value)**2 - config.fp_norm_constraint_coef*(nlp_var.value - mip_var.value)**2
                                                         for nlp_var, mip_var in zip(fp_nlp.MindtPy_utils.discrete_variable_list, solve_data.mip.MindtPy_utils.discrete_variable_list)) <= 0)
        elif config.fp_main_norm == 'L_infinity':
            fp_nlp.norm_constraint = ConstraintList()
            rhs = config.fp_norm_constraint_coef * max(nlp_var.value - mip_var.value for nlp_var, mip_var in zip(
                fp_nlp.MindtPy_utils.discrete_variable_list, solve_data.mip.MindtPy_utils.discrete_variable_list))
            for nlp_var, mip_var in zip(fp_nlp.MindtPy_utils.discrete_variable_list, solve_data.mip.MindtPy_utils.discrete_variable_list):
                fp_nlp.norm_constraint.add(nlp_var - mip_var.value <= rhs)

    MindtPy.fp_nlp_obj = generate_norm2sq_objective_function(
        fp_nlp, solve_data.mip, discrete_only=config.fp_discrete_only)

    MindtPy.cuts.deactivate()
    TransformationFactory('core.relax_integer_vars').apply_to(fp_nlp)
    try:
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
            fp_nlp, tmp=True, ignore_infeasible=False, tolerance=config.constraint_tolerance)
    except ValueError:
        config.logger.warning(
            'infeasibility detected in deactivate_trivial_constraints')
        results = SolverResults()
        results.solver.termination_condition = tc.infeasible
        return fp_nlp, results
    # Solve the NLP
    nlpopt = SolverFactory(config.nlp_solver)
    nlp_args = dict(config.nlp_solver_args)
    set_solver_options(nlpopt, solve_data, config, solver_type='nlp')
    with SuppressInfeasibleWarning():
        with time_code(solve_data.timing, 'fp subproblem'):
            results = nlpopt.solve(
                fp_nlp, tee=config.nlp_solver_tee, **nlp_args)
    return fp_nlp, results


def handle_fp_subproblem_optimal(fp_nlp, solve_data, config):
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

    # if OA-like or fp converged, update Upper bound,
    # add no_good cuts and increasing objective cuts (fp)
    if fp_converged(solve_data, config, discrete_only=config.fp_discrete_only):
        copy_var_list_values(solve_data.mip.MindtPy_utils.variable_list,
                             solve_data.working_model.MindtPy_utils.variable_list,
                             config)
        fixed_nlp, fixed_nlp_results = solve_subproblem(
            solve_data, config)
        if fixed_nlp_results.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            handle_subproblem_optimal(
                fixed_nlp, solve_data, config, fp=True)
        else:
            config.logger.error('Feasibility pump Fixed-NLP is infeasible, something might be wrong. '
                                'There might be a problem with the precisions - the feasibility pump seems to have converged')


def fp_loop(solve_data, config):
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
    while solve_data.fp_iter < config.fp_iteration_limit:

        config.logger.info(
            '---Feasibility Pump Iteration %s---'
            % solve_data.fp_iter)

        solve_data.mip_subiter = 0
        # solve MILP main problem
        feas_main, feas_main_results = solve_main(
            solve_data, config, fp=True)
        if feas_main_results.solver.termination_condition is tc.optimal:
            config.logger.info(
                'FP-MIP %s: Distance-OBJ: %s'
                % (solve_data.fp_iter, value(solve_data.mip.MindtPy_utils.fp_mip_obj)))
        elif feas_main_results.solver.termination_condition is tc.maxTimeLimit:
            config.logger.warning('FP-MIP reaches max TimeLimit')
            solve_data.results.solver.termination_condition = tc.maxTimeLimit
        elif feas_main_results.solver.termination_condition is tc.infeasible:
            config.logger.warning('FP-MIP infeasible')
            no_good_cuts = solve_data.mip.MindtPy_utils.cuts.no_good_cuts
            if no_good_cuts.__len__() > 0:
                no_good_cuts[no_good_cuts.__len__()].deactivate()
            break
        elif feas_main_results.solver.termination_condition is tc.unbounded:
            config.logger.warning('FP-MIP unbounded')
            break
        elif (feas_main_results.solver.termination_condition is tc.other and
              feas_main_results.solution.status is SolutionStatus.feasible):
            config.logger.warning('MILP solver reported feasible solution of FP-MIP, '
                                  'but not guaranteed to be optimal.')
        else:
            config.logger.warning('Unexpected result of FP-MIP')
            break

        # Solve NLP subproblem
        # The constraint linearization happens in the handlers
        fp_nlp, fp_nlp_result = solve_fp_subproblem(
            solve_data, config)

        if fp_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            config.logger.info('FP-NLP %s: Distance-OBJ: %s'
                               % (solve_data.fp_iter, value(fp_nlp.MindtPy_utils.fp_nlp_obj)))
            handle_fp_subproblem_optimal(fp_nlp, solve_data, config)
        elif fp_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
            config.logger.error('Feasibility pump NLP subproblem infeasible')
            solve_data.should_terminate = True
            solve_data.results.solver.status = SolverStatus.error
            return
        elif fp_nlp_result.solver.termination_condition is tc.maxIterations:
            config.logger.error(
                'Feasibility pump NLP subproblem failed to converge within iteration limit.')
            solve_data.should_terminate = True
            solve_data.results.solver.status = SolverStatus.error
            return
        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(fp_nlp_result.solver.termination_condition))
        # Call the NLP post-solve callback
        config.call_after_subproblem_solve(fp_nlp, solve_data)
        solve_data.fp_iter += 1
    solve_data.mip.MindtPy_utils.del_component('fp_mip_obj')

    if config.fp_main_norm == 'L1':
        solve_data.mip.MindtPy_utils.del_component('L1_obj')
    elif config.fp_main_norm == 'L_infinity':
        solve_data.mip.MindtPy_utils.del_component(
            'L_infinity_obj')

    # deactivate the improving_objective_cut
    solve_data.mip.MindtPy_utils.cuts.del_component(
        'improving_objective_cut')
    if not config.fp_transfercuts:
        for c in solve_data.mip.MindtPy_utils.cuts.oa_cuts:
            c.deactivate()
        for c in solve_data.mip.MindtPy_utils.cuts.no_good_cuts:
            c.deactivate()
    if config.fp_projcuts:
        solve_data.working_model.MindtPy_utils.cuts.del_component(
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
    mip_integer_vars = solve_data.mip.MindtPy_utils.discrete_variable_list
    nlp_integer_vars = solve_data.working_model.MindtPy_utils.discrete_variable_list
    orthogonality_cut = sum((nlp_v.value-mip_v.value)*(mip_v-nlp_v.value)
                            for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)) >= 0
    solve_data.mip.MindtPy_utils.cuts.fp_orthogonality_cuts.add(
        orthogonality_cut)
    if config.fp_projcuts:
        orthogonality_cut = sum((nlp_v.value-mip_v.value)*(nlp_v-nlp_v.value)
                                for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)) >= 0
        solve_data.working_model.MindtPy_utils.cuts.fp_orthogonality_cuts.add(
            orthogonality_cut)
