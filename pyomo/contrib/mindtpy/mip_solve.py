# -*- coding: utf-8 -*-
"""Master problem functions."""
from __future__ import division

from pyomo.contrib.gdpopt.util import copy_var_list_values
from pyomo.core import Constraint, Expression, Objective, minimize, value, Var
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing, get_main_elapsed_time, time_code
from pyomo.contrib.gdpopt.mip_solve import distinguish_mip_infeasible_or_unbounded
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.contrib.mindtpy.nlp_solve import (solve_subproblem,
                                             handle_subproblem_optimal, handle_subproblem_infeasible,
                                             handle_subproblem_other_termination, solve_feasibility_subproblem)
from pyomo.contrib.gdpopt.util import copy_var_list_values, identify_variables
from math import copysign
from pyomo.environ import *
from pyomo.core import Constraint, minimize, value
from pyomo.core.expr import current as EXPR
from math import fabs
from pyomo.repn import generate_standard_repn
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.mindtpy.util import generate_norm1_objective_function, generate_norm2sq_objective_function, generate_norm_inf_objective_function

single_tree, single_tree_available = attempt_import(
    'pyomo.contrib.mindtpy.single_tree')


def solve_master(solve_data, config, feas_pump=False):
    """
    This function solves the MIP master problem

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
    master_mip_results: Pyomo results object
        result from solving the master MIP
    """
    if feas_pump:
        # solve_data.mip_iter += 1
        config.logger.info('FP-MIP %s: Solve master problem.' %
                           (solve_data.fp_iter,))
    else:
        solve_data.mip_iter += 1
        config.logger.info('MIP %s: Solve master problem.' %
                           (solve_data.mip_iter,))

    MindtPy = solve_data.mip.MindtPy_utils

    # Set up MILP
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in (1, 0):
            c.deactivate()

    MindtPy.MindtPy_linear_cuts.activate()
    main_objective = next(
        solve_data.mip.component_data_objects(Objective, active=True))
    main_objective.deactivate()

    sign_adjust = 1 if main_objective.sense == minimize else - 1
    MindtPy.del_component('MindtPy_oa_obj')

    if feas_pump:
        if MindtPy.find_component('feas_pump_mip_obj') is not None:
            MindtPy.del_component('feas_pump_mip_obj')
        if config.fp_master_norm == 'L1':
            MindtPy.feas_pump_mip_obj = generate_norm1_objective_function(
                solve_data.mip,
                solve_data.working_model,
                discrete_only=config.fp_discrete_only)
        elif config.fp_master_norm == 'L2':
            MindtPy.feas_pump_mip_obj = generate_norm2sq_objective_function(
                solve_data.mip,
                solve_data.working_model,
                discrete_only=config.fp_discrete_only)
        elif config.fp_master_norm == 'L_infinity':
            MindtPy.feas_pump_mip_obj = generate_norm_inf_objective_function(
                solve_data.mip,
                solve_data.working_model,
                discrete_only=config.fp_discrete_only)
    else:
        if config.add_slack:
            MindtPy.del_component('MindtPy_penalty_expr')

            MindtPy.MindtPy_penalty_expr = Expression(
                expr=sign_adjust * config.OA_penalty_factor * sum(
                    v for v in MindtPy.MindtPy_linear_cuts.slack_vars[...]))

        MindtPy.MindtPy_oa_obj = Objective(
            expr=main_objective.expr +
            (MindtPy.MindtPy_penalty_expr if config.add_slack else 0),
            sense=main_objective.sense)

        if config.use_dual_bound:
            # Delete previously added dual bound constraint
            if MindtPy.MindtPy_linear_cuts.find_component('dual_bound') is not None:
                MindtPy.MindtPy_linear_cuts.del_component('dual_bound')
            if main_objective.sense == minimize:
                MindtPy.MindtPy_linear_cuts.dual_bound = Constraint(
                    expr=main_objective.expr +
                    (MindtPy.MindtPy_penalty_expr if config.add_slack else 0) >= solve_data.LB,
                    doc='Objective function expression should improve on the best found dual bound')
            else:
                MindtPy.MindtPy_linear_cuts.dual_bound = Constraint(
                    expr=main_objective.expr +
                    (MindtPy.MindtPy_penalty_expr if config.add_slack else 0) <= solve_data.UB,
                    doc='Objective function expression should improve on the best found dual bound')

    # Deactivate extraneous IMPORT/EXPORT suffixes
    if config.nlp_solver == 'ipopt':
        getattr(solve_data.mip, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(solve_data.mip, 'ipopt_zU_out', _DoNothing()).deactivate()

    masteropt = SolverFactory(config.mip_solver)
    # determine if persistent solver is called.
    if isinstance(masteropt, PersistentSolver):
        masteropt.set_instance(solve_data.mip, symbolic_solver_labels=True)
    if config.single_tree:
        # Configuration of lazy callback
        lazyoa = masteropt._solver_model.register_callback(
            single_tree.LazyOACallback_cplex)
        # pass necessary data and parameters to lazyoa
        lazyoa.master_mip = solve_data.mip
        lazyoa.solve_data = solve_data
        lazyoa.config = config
        lazyoa.opt = masteropt
        masteropt._solver_model.set_warning_stream(None)
        masteropt._solver_model.set_log_stream(None)
        masteropt._solver_model.set_error_stream(None)
        masteropt.options['timelimit'] = config.time_limit
    if config.threads > 0:
        masteropt.options["threads"] = config.threads
    mip_args = dict(config.mip_solver_args)
    elapsed = get_main_elapsed_time(solve_data.timing)
    remaining = int(max(config.time_limit - elapsed, 1))
    if config.mip_solver == 'gams':
        mip_args['add_options'] = mip_args.get('add_options', [])
        mip_args['add_options'].append('option optcr=0.001;')
        mip_args['add_options'].append('option reslim=%s;' % remaining)
    # elif config.mip_solver == 'glpk':
    #     masteropt.options['timelimit'] = remaining
    try:
        with time_code(solve_data.timing, 'mip'):
            master_mip_results = masteropt.solve(
                solve_data.mip, tee=config.mip_solver_tee, **mip_args)

        # if config.single_tree is False and config.add_nogood_cuts is False:

        if master_mip_results.solver.termination_condition is tc.optimal:
            if config.single_tree and config.add_nogood_cuts is False:
                if main_objective.sense == minimize:
                    solve_data.LB = max(
                        master_mip_results.problem.lower_bound, solve_data.LB)
                    solve_data.LB_progress.append(solve_data.LB)
                else:
                    solve_data.UB = min(
                        master_mip_results.problem.upper_bound, solve_data.UB)
                    solve_data.UB_progress.append(solve_data.UB)

        elif master_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            master_mip_results, _ = distinguish_mip_infeasible_or_unbounded(
                solve_data.mip, config)
        return solve_data.mip, master_mip_results
    except ValueError:
        config.logger.warning("ValueError: Cannot load a SolverResults object with bad status: error. "
                              "MIP solver failed. This usually happens in the single-tree GOA algorithm. "
                              "Nogood cuts are added and GOA algorithm doesn't converge within the time limit. "
                              "No integer solution is found, so the cplex solver will report an error status. ")
        return None, None


# The following functions deal with handling the solution we get from the above MIP solver function


def handle_master_optimal(master_mip, solve_data, config):
    """
    This function copies the result from 'solve_master' to the working model and updates the upper/lower bound. This
    function is called after an optimal solution is found for the master problem.

    Parameters
    ----------
    master_mip: Pyomo model
        the MIP master problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # proceed. Just need integer values
    MindtPy = master_mip.MindtPy_utils
    main_objective = next(
        master_mip.component_data_objects(Objective, active=True))
    # check if the value of binary variable is valid
    for var in MindtPy.variable_list:
        if var.value is None and var.is_integer():
            config.logger.warning(
                "Integer variable {} not initialized. It is set to it's lower bound".format(var.name))
            var.value = var.lb  # nlp_var.bounds[0]
    # warm start for the nlp subproblem
    copy_var_list_values(
        master_mip.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)

    if main_objective.sense == minimize:
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


def handle_master_other_conditions(master_mip, master_mip_results, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem (given any of a few
    edge conditions, such as if the solution is neither infeasible nor optimal).

    Parameters
    ----------
    master_mip: Pyomo model
        the MIP master problem
    master_mip_results: Pyomo results object
        result from solving the MIP problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    if master_mip_results.solver.termination_condition is tc.infeasible:
        handle_master_infeasible(master_mip, solve_data, config)
    elif master_mip_results.solver.termination_condition is tc.unbounded:
        handle_master_unbounded(master_mip, solve_data, config)
    elif master_mip_results.solver.termination_condition is tc.maxTimeLimit:
        handle_master_max_timelimit(master_mip, solve_data, config)
    elif (master_mip_results.solver.termination_condition is tc.other and
          master_mip_results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        MindtPy = master_mip.MindtPy_utils
        config.logger.info(
            'MILP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
        copy_var_list_values(
            master_mip.MindtPy_utils.variable_list,
            solve_data.working_model.MindtPy_utils.variable_list,
            config)
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
    else:
        raise ValueError(
            'MindtPy unable to handle MILP master termination condition '
            'of %s. Solver message: %s' %
            (master_mip_results.solver.termination_condition, master_mip_results.solver.message))


def handle_master_infeasible(master_mip, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem given an infeasible solution.

    Parameters
    ----------
    master_mip: Pyomo model
        the MIP master problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    config.logger.info(
        'MILP master problem is infeasible. '
        'Problem may have no more feasible '
        'binary configurations.')
    if solve_data.mip_iter == 1:
        config.logger.warning(
            'MindtPy initialization may have generated poor '
            'quality cuts.')
    # TODO nogood cuts for single tree case
    # set optimistic bound to infinity
    main_objective = next(
        master_mip.component_data_objects(Objective, active=True))
    if main_objective.sense == minimize:
        solve_data.LB_progress.append(solve_data.LB)
    else:
        solve_data.UB_progress.append(solve_data.UB)
    config.logger.info(
        'MindtPy exiting due to MILP master problem infeasibility.')
    if solve_data.results.solver.termination_condition is None:
        if solve_data.mip_iter == 0:
            solve_data.results.solver.termination_condition = tc.infeasible
        else:
            solve_data.results.solver.termination_condition = tc.feasible


def handle_master_max_timelimit(master_mip, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem given that solving the
    MIP takes too long.

    Parameters
    ----------
    master_mip: Pyomo model
        the MIP master problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # TODO check that status is actually ok and everything is feasible
    MindtPy = master_mip.MindtPy_utils
    config.logger.info(
        'Unable to optimize MILP master problem '
        'within time limit. '
        'Using current solver feasible solution.')
    copy_var_list_values(
        master_mip.MindtPy_utils.variable_list,
        solve_data.working_model.MindtPy_utils.variable_list,
        config)
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


def handle_master_unbounded(master_mip, solve_data, config):
    """
    This function handles the result of the latest iteration of solving the MIP problem given an unbounded solution
    due to the relaxation.

    Parameters
    ----------
    master_mip: Pyomo model
        the MIP master problem
    solve_data: MindtPy Data Container
        data container that holds solve-instance data
    config: ConfigBlock
        contains the specific configurations for the algorithm
    """
    # Solution is unbounded. Add an arbitrary bound to the objective and resolve.
    # This occurs when the objective is nonlinear. The nonlinear objective is moved
    # to the constraints, and deactivated for the linear master problem.
    MindtPy = master_mip.MindtPy_utils
    config.logger.warning(
        'Master MILP was unbounded. '
        'Resolving with arbitrary bound values of (-{0:.10g}, {0:.10g}) on the objective. '
        'You can change this bound with the option obj_bound.'.format(config.obj_bound))
    main_objective = next(
        master_mip.component_data_objects(Objective, active=True))
    MindtPy.objective_bound = Constraint(
        expr=(-config.obj_bound, main_objective.expr, config.obj_bound))
    with SuppressInfeasibleWarning():
        opt = SolverFactory(config.mip_solver)
        if isinstance(opt, PersistentSolver):
            opt.set_instance(master_mip)
        master_mip_results = opt.solve(
            master_mip, tee=config.mip_solver_tee, **config.mip_solver_args)
