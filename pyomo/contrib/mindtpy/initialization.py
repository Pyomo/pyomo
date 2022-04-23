#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Initialization functions."""
from __future__ import division
from pyomo.contrib.gdpopt.util import (SuppressInfeasibleWarning, _DoNothing,
                                       copy_var_list_values, get_main_elapsed_time)
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_affine_cuts
from pyomo.contrib.mindtpy.nlp_solve import solve_subproblem
from pyomo.contrib.mindtpy.util import calc_jacobians, set_solver_options, update_dual_bound, add_var_bound, get_integer_solution, update_suboptimal_dual_bound
from pyomo.core import (ConstraintList, Objective,
                        TransformationFactory, maximize, minimize,
                        value, Var)
from pyomo.opt import SolverFactory, TerminationCondition as tc
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.contrib.mindtpy.nlp_solve import solve_subproblem, handle_nlp_subproblem_tc
from pyomo.contrib.mindtpy.feasibility_pump import fp_loop


def MindtPy_initialize_main(solve_data, config):
    """Initializes the decomposition algorithm and creates the main MIP/MILP problem.

    This function initializes the decomposition problem, which includes generating the
    initial cuts required to build the main MIP.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    # if single tree is activated, we need to add bounds for unbounded variables in nonlinear constraints to avoid unbounded main problem.
    if config.single_tree:
        add_var_bound(solve_data, config)

    m = solve_data.mip = solve_data.working_model.clone()
    next(solve_data.mip.component_data_objects(
        Objective, active=True)).deactivate()

    MindtPy = m.MindtPy_utils
    if config.calculate_dual:
        m.dual.deactivate()

    if config.init_strategy == 'FP':
        MindtPy.cuts.fp_orthogonality_cuts = ConstraintList(
            doc='Orthogonality cuts in feasibility pump')
        if config.fp_projcuts:
            solve_data.working_model.MindtPy_utils.cuts.fp_orthogonality_cuts = ConstraintList(
                doc='Orthogonality cuts in feasibility pump')
    if config.strategy == 'OA' or config.init_strategy == 'FP':
        calc_jacobians(solve_data, config)  # preload jacobians
        MindtPy.cuts.oa_cuts = ConstraintList(doc='Outer approximation cuts')
    elif config.strategy == 'ECP':
        calc_jacobians(solve_data, config)  # preload jacobians
        MindtPy.cuts.ecp_cuts = ConstraintList(doc='Extended Cutting Planes')
    elif config.strategy == 'GOA':
        MindtPy.cuts.aff_cuts = ConstraintList(doc='Affine cuts')
    # elif config.strategy == 'PSC':
    #     detect_nonlinear_vars(solve_data, config)
    #     MindtPy.cuts.psc_cuts = ConstraintList(
    #         doc='Partial surrogate cuts')
    # elif config.strategy == 'GBD':
    #     MindtPy.cuts.gbd_cuts = ConstraintList(
    #         doc='Generalized Benders cuts')

    # Set default initialization_strategy
    if config.init_strategy is None:
        if config.strategy in {'OA', 'GOA'}:
            config.init_strategy = 'rNLP'
        else:
            config.init_strategy = 'max_binary'

    config.logger.info(
        '{} is the initial strategy being used.'
        '\n'.format(config.init_strategy))
    config.logger.info(
        ' ===============================================================================================')
    config.logger.info(
        ' {:>9} | {:>15} | {:>15} | {:>12} | {:>12} | {:^7} | {:>7}\n'.format('Iteration', 'Subproblem Type', 'Objective Value', 'Primal Bound',
                                                                              'Dual Bound', ' Gap ', 'Time(s)'))
    # Do the initialization
    if config.init_strategy == 'rNLP':
        init_rNLP(solve_data, config)
    elif config.init_strategy == 'max_binary':
        init_max_binaries(solve_data, config)
    elif config.init_strategy == 'initial_binary':
        solve_data.curr_int_sol = get_integer_solution(
            solve_data.working_model)
        solve_data.integer_list.append(solve_data.curr_int_sol)
        if config.strategy != 'ECP':
            fixed_nlp, fixed_nlp_result = solve_subproblem(solve_data, config)
            handle_nlp_subproblem_tc(
                fixed_nlp, fixed_nlp_result, solve_data, config)
    elif config.init_strategy == 'FP':
        init_rNLP(solve_data, config)
        fp_loop(solve_data, config)


def init_rNLP(solve_data, config):
    """Initialize the problem by solving the relaxed NLP and then store the optimal variable
    values obtained from solving the rNLP.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.

    Raises
    ------
    ValueError
        MindtPy unable to handle the termination condition of the relaxed NLP.
    """
    m = solve_data.working_model.clone()
    config.logger.debug(
        'Relaxed NLP: Solve relaxed integrality')
    MindtPy = m.MindtPy_utils
    TransformationFactory('core.relax_integer_vars').apply_to(m)
    nlp_args = dict(config.nlp_solver_args)
    nlpopt = SolverFactory(config.nlp_solver)
    set_solver_options(nlpopt, solve_data, config, solver_type='nlp')
    with SuppressInfeasibleWarning():
        results = nlpopt.solve(m, tee=config.nlp_solver_tee, **nlp_args)
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond in {tc.optimal, tc.feasible, tc.locallyOptimal}:
        main_objective = MindtPy.objective_list[-1]
        if subprob_terminate_cond == tc.optimal:
            update_dual_bound(solve_data, value(main_objective.expr))
        else:
            config.logger.info(
                'relaxed NLP is not solved to optimality.')
            update_suboptimal_dual_bound(solve_data, results)
        dual_values = list(
            m.dual[c] for c in MindtPy.constraint_list) if config.calculate_dual else None
        config.logger.info(solve_data.log_formatter.format('-', 'Relaxed NLP', value(main_objective.expr),
                                                           solve_data.primal_bound, solve_data.dual_bound, solve_data.rel_gap,
                                                           get_main_elapsed_time(solve_data.timing)))
        # Add OA cut
        if config.strategy in {'OA', 'GOA', 'FP'}:
            copy_var_list_values(m.MindtPy_utils.variable_list,
                                 solve_data.mip.MindtPy_utils.variable_list,
                                 config, ignore_integrality=True)
            if config.init_strategy == 'FP':
                copy_var_list_values(m.MindtPy_utils.variable_list,
                                     solve_data.working_model.MindtPy_utils.variable_list,
                                     config, ignore_integrality=True)
            if config.strategy in {'OA', 'FP'}:
                add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
            elif config.strategy == 'GOA':
                add_affine_cuts(solve_data, config)
            for var in solve_data.mip.MindtPy_utils.discrete_variable_list:
                # We don't want to trigger the reset of the global stale
                # indicator, so we will set this variable to be "stale",
                # knowing that set_value will switch it back to "not
                # stale"
                var.stale = True
                var.set_value(int(round(var.value)), skip_validation=True)
    elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
        # TODO fail? try something else?
        config.logger.info(
            'Initial relaxed NLP problem is infeasible. '
            'Problem may be infeasible.')
    elif subprob_terminate_cond is tc.maxTimeLimit:
        config.logger.info(
            'NLP subproblem failed to converge within time limit.')
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
    elif subprob_terminate_cond is tc.maxIterations:
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
    else:
        raise ValueError(
            'MindtPy unable to handle relaxed NLP termination condition '
            'of %s. Solver message: %s' %
            (subprob_terminate_cond, results.solver.message))


def init_max_binaries(solve_data, config):
    """Modifies model by maximizing the number of activated binary variables.

    Note - The user would usually want to call solve_subproblem after an invocation 
    of this function.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.

    Raises
    ------
    ValueError
        MILP main problem is infeasible.
    ValueError
        MindtPy unable to handle the termination condition of the MILP main problem.
    """
    m = solve_data.working_model.clone()
    if config.calculate_dual:
        m.dual.deactivate()
    MindtPy = m.MindtPy_utils
    solve_data.mip_subiter += 1
    config.logger.debug(
        'Initialization: maximize value of binaries')
    for c in MindtPy.nonlinear_constraint_list:
        c.deactivate()
    objective = next(m.component_data_objects(Objective, active=True))
    objective.deactivate()
    binary_vars = (v for v in m.MindtPy_utils.discrete_variable_list
                   if v.is_binary() and not v.fixed)
    MindtPy.max_binary_obj = Objective(
        expr=sum(v for v in binary_vars), sense=maximize)

    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    mipopt = SolverFactory(config.mip_solver)
    if isinstance(mipopt, PersistentSolver):
        mipopt.set_instance(m)
    mip_args = dict(config.mip_solver_args)
    set_solver_options(mipopt, solve_data, config, solver_type='mip')
    results = mipopt.solve(m, tee=config.mip_solver_tee, **mip_args)

    solve_terminate_cond = results.solver.termination_condition
    if solve_terminate_cond is tc.optimal:
        copy_var_list_values(
            MindtPy.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
        config.logger.info(solve_data.log_formatter.format('-', 'Max binary MILP', value(MindtPy.max_binary_obj.expr),
                                                           solve_data.primal_bound, solve_data.dual_bound, solve_data.rel_gap,
                                                           get_main_elapsed_time(solve_data.timing)))
    elif solve_terminate_cond is tc.infeasible:
        raise ValueError(
            'MILP main problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.')
    elif solve_terminate_cond is tc.maxTimeLimit:
        config.logger.info(
            'NLP subproblem failed to converge within time limit.')
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
    elif solve_terminate_cond is tc.maxIterations:
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
    else:
        raise ValueError(
            'MindtPy unable to handle MILP main termination condition '
            'of %s. Solver message: %s' %
            (solve_terminate_cond, results.solver.message))
