"""Master problem functions."""
from __future__ import division

from pyomo.contrib.mindtpy.initialization import init_max_binaries
from pyomo.contrib.mindtpy.util import copy_values
from pyomo.core import Constraint, Expression, Objective, minimize, value
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing
from pyomo.contrib.gdpopt.mip_solve import distinguish_mip_infeasible_or_unbounded


def solve_OA_master(solve_data, config):
    solve_data.mip_iter += 1
    m = solve_data.mip.clone()
    MindtPy = m.MindtPy_utils
    config.logger.info(
        'MIP %s: Solve master problem.' %
        (solve_data.mip_iter,))
    # Set up MILP
    for c in MindtPy.nonlinear_constraints:
        c.deactivate()

    MindtPy.MindtPy_linear_cuts.activate()
    MindtPy.objective.deactivate()

    sign_adjust = 1 if MindtPy.objective.sense == minimize else -1
    MindtPy.MindtPy_penalty_expr = Expression(
        expr=sign_adjust * config.OA_penalty_factor * sum(
            v for v in MindtPy.MindtPy_linear_cuts.slack_vars[...]))

    MindtPy.MindtPy_oa_obj = Objective(
        expr=MindtPy.objective.expr + MindtPy.MindtPy_penalty_expr,
        sense=MindtPy.objective.sense)

    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    # m.pprint() #print oa master problem for debugging
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.mip_solver).solve(
            m, **config.mip_solver_args)
    master_terminate_cond = results.solver.termination_condition
    if master_terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        results, master_terminate_cond = distinguish_mip_infeasible_or_unbounded(
            m, config)

    # Process master problem result
    if master_terminate_cond is tc.optimal:
        # proceed. Just need integer values
        copy_values(m, solve_data.working_model, config)

        if MindtPy.objective.sense == minimize:
            solve_data.LB = max(
                value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
               solve_data.LB, solve_data.UB))
    elif master_terminate_cond is tc.infeasible:
        print('MILP master problem is infeasible. '
              'Problem may have no more feasible binary combinations.')
        if solve_data.mip_iter == 1:
            print('MindtPy initialization may have generated poor '
                  'quality cuts.')
    elif master_terminate_cond is tc.maxTimeLimit:
        # TODO check that status is actually ok and everything is feasible
        config.logger.info(
            'Unable to optimize MILP master problem '
            'within time limit. '
            'Using current solver feasible solution.')
        copy_values(m, solve_data.working_model)
        if MindtPy.obj.sense == minimize:
            solve_data.LB = max(
                value(MindtPy.obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                value(MindtPy.obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.obj.expr),
               solve_data.LB, solve_data.UB))
    elif (master_terminate_cond is tc.other and
            results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        config.logger.info(
            'MILP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
        copy_values(m, solve_data.working_model)
        if MindtPy.obj.sense == minimize:
            solve_data.LB = max(
                value(MindtPy.MindtPy_oa_obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(
                value(MindtPy.MindtPy_oa_obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.MindtPy_oa_obj.expr),
               solve_data.LB, solve_data.UB))
    elif master_terminate_cond is tc.infeasible:
        config.logger.info(
            'MILP master problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.')
        if solve_data.mip_iter == 1:
            config.logger.warn(
                'MindtPy initialization may have generated poor '
                'quality cuts.')
        # set optimistic bound to infinity
        if MindtPy.obj.sense == minimize:
            solve_data.LB = float('inf')
            solve_data.LB_progress.append(solve_data.UB)
        else:
            solve_data.UB = float('-inf')
            solve_data.UB_progress.append(solve_data.UB)
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of %s. Solver message: %s' %
            (master_terminate_cond, results.solver.message))

    # Call the MILP post-solve callback
    config.call_after_master_solve(m, solve_data)


def solve_ECP_master(solve_data, config):
    solve_data.mip_iter += 1
    m = solve_data.working_model.clone()
    MindtPy = m.MindtPy_utils

    feas_sol = 0
    config.logger.info(
        'MIP %s: Solve master problem.' %
        (solve_data.mip_iter,))
    # Set up MILP
    for c in MindtPy.nonlinear_constraints:
        c.deactivate()
    MindtPy.MindtPy_linear_cuts.activate()

    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    with SuppressInfeasibleWarning():
        results = SolverFactory(config.mip_solver).solve(
            m, **config.mip_solver_args)
    master_terminate_cond = results.solver.termination_condition
    if master_terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        results, master_terminate_cond = distinguish_mip_infeasible_or_unbounded(
            m, config)
    for c in MindtPy.nonlinear_constraints:
        c.activate()
        MindtPy.MindtPy_linear_cuts.deactivate()

    # Process master problem result
    if master_terminate_cond is tc.optimal:
        # proceed. Just need integer values
        copy_values(m, solve_data.working_model, config)

        if all(
            (0 if c.upper is None
             else (value(c.body) - c.upper)) +
            (0 if c.lower is None
             else (c.lower - value(c.body)))
                < config.ECP_tolerance
                for c in MindtPy.nonlinear_constraints):
            solve_data.best_solution_found = m.clone()
            feas_sol = 1
            print('ECP has found a feasible solution within a {} tolerance'
                  .format(config.ECP_tolerance))
        if MindtPy.obj.sense == minimize:
            solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
            if feas_sol == 1:
                solve_data.UB = value(MindtPy.obj.expr)
        else:
            solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
            if feas_sol == 1:
                solve_data.LB = value(MindtPy.obj.expr)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.obj.expr),
               solve_data.LB, solve_data.UB))
    elif master_terminate_cond is tc.infeasible:
        print('MILP master problem is infeasible. '
              'Problem may have no more feasible binary combinations.')
        if solve_data.mip_iter == 1:
            print('MindtPy initialization may have generated poor '
                  'quality cuts.')
        # set optimistic bound to infinity
        if MindtPy.obj.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of {}. Solver message: {}'.format(
                master_terminate_cond, results.solver.message))

    # Call the MILP post-solve callback
    config.call_after_master_solve(m, solve_data)


def solve_PSC_master(solve_data, config):
    solve_data.mip_iter += 1
    m = solve_data.working_model.clone()
    MindtPy = m.MindtPy_utils

    config.logger.info(
        'MIP %s: Solve master problem.' %
        (solve_data.mip_iter,))
    # Set up MILP
    for c in MindtPy.nonlinear_constraints:
        c.deactivate()
    MindtPy.MindtPy_linear_cuts.activate()
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
    # m.pprint() #print psc master problem for debugging
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.mip_solver).solve(
            m, **config.mip_solver_args)
    for c in MindtPy.nonlinear_constraints:
        c.activate()
    MindtPy.MindtPy_linear_cuts.deactivate()
    getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

    # Process master problem result
    master_terminate_cond = results.solver.termination_condition
    if master_terminate_cond is tc.optimal:
        # proceed. Just need integer values
        copy_values(m, solve_data.working_model, config)

        if MindtPy.obj.sense == minimize:
            solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.obj.expr),
               solve_data.LB, solve_data.UB))
    elif master_terminate_cond is tc.infeasible:
        print('MILP master problem is infeasible. '
              'Problem may have no more feasible binary combinations.')
        if solve_data.mip_iter == 1:
            print('MindtPy initialization may have generated poor '
                  'quality cuts.')
        # set optimistic bound to infinity
        if MindtPy.obj.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of {}. Solver message: {}'.format(
                master_terminate_cond, results.solver.message))

    # Call the MILP post-solve callback
    config.call_after_master_solve(m, solve_data)


def solve_GBD_master(solve_data, config, leave_linear_active=True):
    solve_data.mip_iter += 1
    m = solve_data.working_model.clone()
    MindtPy = m.MindtPy_utils

    config.logger.info(
        'MIP %s: Solve master problem.' %
        (solve_data.mip_iter,))
    if not leave_linear_active:
        # Deactivate all constraints except those in MindtPy_linear_cuts
        _MindtPy_linear_cuts = set(
            c for c in MindtPy.MindtPy_linear_cuts.component_data_objects(
                ctype=Constraint, descend_into=True))
        to_deactivate = set(c for c in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True)
            if c not in _MindtPy_linear_cuts)
        for c in to_deactivate:
            c.deactivate()
    else:
        for c in MindtPy.nonlinear_constraints:
            c.deactivate()
    MindtPy.MindtPy_linear_cuts.activate()
    # m.MindtPy_objective_expr.activate() # This activation will be deleted
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
    # m.pprint() #print gbd master problem for debugging
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.mip_solver).solve(
            m, **config.mip_solver_args)
    master_terminate_cond = results.solver.termination_condition
    if master_terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it is infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        results, master_terminate_cond = distinguish_mip_infeasible_or_unbounded(
            m, config)
    if not leave_linear_active:
        for c in to_deactivate:
            c.activate()
    else:
        for c in MindtPy.nonlinear_constraints:
            c.activate()
    MindtPy.MindtPy_linear_cuts.deactivate()
    getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

    # Process master problem result
    if master_terminate_cond is tc.optimal:
        # proceed. Just need integer values
        copy_values(m, solve_data.working_model, config)

        if MindtPy.obj.sense == minimize:
            solve_data.LB = max(value(MindtPy.obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(value(MindtPy.obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'MIP %s: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.mip_iter, value(MindtPy.obj.expr),
               solve_data.LB, solve_data.UB))
    elif master_terminate_cond is tc.infeasible:
        print('MILP master problem is infeasible. '
              'Problem may have no more feasible binary configurations.')
        if solve_data.mip_iter == 1:
            print('MindtPy initialization may have generated poor '
                  'quality cuts.')
        # set optimistic bound to infinity
        if MindtPy.obj.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
    elif master_terminate_cond is tc.unbounded:
        print('MILP master problem is unbounded. ')
        # Change the integer values to something new, re-solve.
        MindtPy.MindtPy_linear_cuts.activate()
        MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.activate()
        init_max_binaries()
        MindtPy.MindtPy_linear_cuts.deactivate()
        MindtPy.MindtPy_linear_cuts.feasible_integer_cuts.deactivate()
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of {}. Solver message: {}'.format(
                master_terminate_cond, results.solver.message))

    #
    # MindtPy.MindtPy_linear_cuts.deactivate()
    # Call the MILP post-solve callback
    config.call_after_master_solve(m, solve_data)
