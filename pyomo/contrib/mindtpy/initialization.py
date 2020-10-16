# -*- coding: utf-8 -*-
"""Initialization functions."""
from __future__ import division

from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing, copy_var_list_values, get_main_elapsed_time
from pyomo.contrib.mindtpy.cut_generation import (
    add_oa_cuts, add_affine_cuts, add_objective_linearization,
)
from pyomo.contrib.mindtpy.nlp_solve import solve_subproblem
from pyomo.contrib.mindtpy.util import (calc_jacobians)
from pyomo.core import (ConstraintList, Objective,
                        TransformationFactory, maximize, minimize, value, Var, Constraint)
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.contrib.mindtpy.nlp_solve import (solve_subproblem,
                                             handle_subproblem_optimal, handle_subproblem_infeasible,
                                             handle_subproblem_other_termination)
from pyomo.contrib.mindtpy.util import var_bound_add
from pyomo.contrib.mindtpy.cut_generation import (add_oa_cuts, add_ecp_cuts)
import math
from pyomo.contrib.mindtpy.feasibility_pump import feas_pump_loop


def MindtPy_initialize_master(solve_data, config):
    """
    Initializes the decomposition algorithm and creates the master MIP/MILP problem.

    This function initializes the decomposition problem, which includes generating the initial cuts required to
    build the master MIP/MILP

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # if single tree is activated, we need to add bounds for unbounded variables in nonlinear constraints to avoid unbounded master problem.
    if config.single_tree:
        var_bound_add(solve_data, config)

    m = solve_data.mip = solve_data.working_model.clone()
    MindtPy = m.MindtPy_utils
    if config.use_dual:
        m.dual.deactivate()

    if config.init_strategy == 'feas_pump':
        MindtPy.MindtPy_linear_cuts.fp_orthogonality_cuts = ConstraintList(
            doc='Orthogonality cuts in feasibility pump')
        if config.fp_projcuts:
            solve_data.working_model.MindtPy_utils.MindtPy_linear_cuts.fp_orthogonality_cuts = ConstraintList(
                doc='Orthogonality cuts in feasibility pump')
    if config.strategy == 'OA' or config.init_strategy == 'feas_pump':
        calc_jacobians(solve_data, config)  # preload jacobians
        MindtPy.MindtPy_linear_cuts.oa_cuts = ConstraintList(
            doc='Outer approximation cuts')
    elif config.strategy == 'ECP':
        calc_jacobians(solve_data, config)  # preload jacobians
        MindtPy.MindtPy_linear_cuts.ecp_cuts = ConstraintList(
            doc='Extended Cutting Planes')
    # elif config.strategy == 'PSC':
    #     detect_nonlinear_vars(solve_data, config)
    #     MindtPy.MindtPy_linear_cuts.psc_cuts = ConstraintList(
    #         doc='Partial surrogate cuts')
    # elif config.strategy == 'GBD':
    #     MindtPy.MindtPy_linear_cuts.gbd_cuts = ConstraintList(
    #         doc='Generalized Benders cuts')

    # Set default initialization_strategy
    if config.init_strategy is None:
        if config.strategy in {'OA', 'GOA'}:
            config.init_strategy = 'rNLP'
        else:
            config.init_strategy = 'max_binary'

    config.logger.info(
        '{} is the initial strategy being used.'
        '\n'.format(
            config.init_strategy))
    # Do the initialization
    if config.init_strategy == 'rNLP':
        init_rNLP(solve_data, config)
    elif config.init_strategy == 'max_binary':
        init_max_binaries(solve_data, config)
    elif config.init_strategy == 'initial_binary':
        if config.strategy != 'ECP':
            fixed_nlp, fixed_nlp_result = solve_subproblem(
                solve_data, config)
            if fixed_nlp_result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
                handle_subproblem_optimal(fixed_nlp, solve_data, config)
            elif fixed_nlp_result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
                handle_subproblem_infeasible(fixed_nlp, solve_data, config)
            else:
                handle_subproblem_other_termination(fixed_nlp, fixed_nlp_result.solver.termination_condition,
                                                    solve_data, config)
    elif config.init_strategy == 'feas_pump':
        init_rNLP(solve_data, config)
        feas_pump_loop(solve_data, config)


def init_rNLP(solve_data, config):
    """
    Initialize the problem by solving the relaxed NLP (fixed binary variables) and then store the optimal variable
    values obtained from solving the rNLP

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    m = solve_data.working_model.clone()
    config.logger.info(
        "Relaxed NLP: Solve relaxed integrality")
    MindtPy = m.MindtPy_utils
    TransformationFactory('core.relax_integer_vars').apply_to(m)
    nlp_args = dict(config.nlp_solver_args)
    elapsed = get_main_elapsed_time(solve_data.timing)
    remaining = int(max(config.time_limit - elapsed, 1))
    if config.nlp_solver == 'gams':
        nlp_args['add_options'] = nlp_args.get('add_options', [])
        nlp_args['add_options'].append('option reslim=%s;' % remaining)
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.nlp_solver).solve(
            m, tee=config.nlp_solver_tee, **nlp_args)
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond in {tc.optimal, tc.feasible, tc.locallyOptimal}:
        if subprob_terminate_cond in {tc.feasible, tc.locallyOptimal}:
            config.logger.info(
                'relaxed NLP is not solved to optimality.')
        main_objective = next(m.component_data_objects(Objective, active=True))
        nlp_solution_values = list(v.value for v in MindtPy.variable_list)
        dual_values = list(
            m.dual[c] for c in MindtPy.constraint_list) if config.use_dual else None
        # Add OA cut
        # This covers the case when the Lower bound does not exist.
        if main_objective.sense == minimize and not math.isnan(results['Problem'][0]['Lower bound']):
            solve_data.LB = results['Problem'][0]['Lower bound']
        elif not math.isnan(results['Problem'][0]['Upper bound']):
            solve_data.UB = results['Problem'][0]['Upper bound']
        config.logger.info(
            'Relaxed NLP: OBJ: %s  LB: %s  UB: %s'
            % (value(main_objective.expr), solve_data.LB, solve_data.UB))
        if config.strategy in {'OA', 'GOA', 'feas_pump'}:
            copy_var_list_values(m.MindtPy_utils.variable_list,
                                 solve_data.mip.MindtPy_utils.variable_list,
                                 config, ignore_integrality=True)
            if config.init_strategy == 'feas_pump':
                # TODO：remove here
                copy_var_list_values(m.MindtPy_utils.variable_list,
                                     solve_data.working_model.MindtPy_utils.variable_list,
                                     config, ignore_integrality=True)
            if config.strategy == 'OA':
                add_oa_cuts(solve_data.mip, dual_values, solve_data, config)
            elif config.strategy == 'GOA':
                add_affine_cuts(solve_data, config)
            # TODO check if value of the binary or integer varibles is 0/1 or integer value.
            for var in solve_data.mip.component_data_objects(ctype=Var):
                if var.is_integer():
                    var.value = int(round(var.value))
    elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
        # TODO fail? try something else?
        config.logger.info(
            'Initial relaxed NLP problem is infeasible. '
            'Problem may be infeasible.')
    elif subprob_terminate_cond is tc.maxTimeLimit:
        config.logger.info(
            'NLP subproblem failed to converge within time limit.')
    elif subprob_terminate_cond is tc.maxIterations:
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
    else:
        raise ValueError(
            'MindtPy unable to handle relaxed NLP termination condition '
            'of %s. Solver message: %s' %
            (subprob_terminate_cond, results.solver.message))


def init_max_binaries(solve_data, config):
    """
    Modifies model by maximizing the number of activated binary variables

    Note - The user would usually want to call solve_subproblem after an
    invocation of this function.

    Parameters
    ----------
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    m = solve_data.working_model.clone()
    if config.use_dual:
        m.dual.deactivate()
    MindtPy = m.MindtPy_utils
    solve_data.mip_subiter += 1
    config.logger.info(
        "MILP %s: maximize value of binaries" %
        (solve_data.mip_iter))
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in (1, 0):
            c.deactivate()
    objective = next(m.component_data_objects(Objective, active=True))
    objective.deactivate()
    binary_vars = (
        v for v in m.component_data_objects(ctype=Var)
        if v.is_binary() and not v.fixed)
    MindtPy.MindtPy_max_binary_obj = Objective(
        expr=sum(v for v in binary_vars), sense=maximize)

    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    opt = SolverFactory(config.mip_solver)
    if isinstance(opt, PersistentSolver):
        opt.set_instance(m)
    mip_args = dict(config.mip_solver_args)
    elapsed = get_main_elapsed_time(solve_data.timing)
    remaining = int(max(config.time_limit - elapsed, 1))
    if config.mip_solver == 'gams':
        mip_args['add_options'] = mip_args.get('add_options', [])
        mip_args['add_options'].append('option optcr=0.001;')
    results = opt.solve(m, tee=config.mip_solver_tee, **mip_args)

    solve_terminate_cond = results.solver.termination_condition
    if solve_terminate_cond is tc.optimal:
        copy_var_list_values(
            MindtPy.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)

        pass  # good
    elif solve_terminate_cond is tc.infeasible:
        raise ValueError(
            'MILP master problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.')
    elif subprob_terminate_cond is tc.maxTimeLimit:
        config.logger.info(
            'NLP subproblem failed to converge within time limit.')
    elif subprob_terminate_cond is tc.maxIterations:
        config.logger.info(
            'NLP subproblem failed to converge within iteration limit.')
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of %s. Solver message: %s' %
            (solve_terminate_cond, results.solver.message))
