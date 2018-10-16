"""Functions for solving the master problem."""

from __future__ import division

from copy import deepcopy

from pyomo.contrib.gdpopt.data_class import MasterProblemResult
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, _DoNothing
from pyomo.core import (Block, Expression, Objective, TransformationFactory,
                        Var, minimize, value)
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory


def solve_linear_GDP(linear_GDP_model, solve_data, config):
    """Solves the linear GDP model and attempts to resolve solution issues."""
    m = linear_GDP_model
    GDPopt = m.GDPopt_utils
    # Transform disjunctions
    TransformationFactory('gdp.bigm').apply_to(m)

    preprocessing_transformations = [
        # Propagate variable bounds
        'contrib.propagate_eq_var_bounds',
        # Detect fixed variables
        'contrib.detect_fixed_vars',
        # Propagate fixed variables
        'contrib.propagate_fixed_vars',
        # Remove zero terms in linear expressions
        'contrib.remove_zero_terms',
        # Remove terms in equal to zero summations
        'contrib.propagate_zero_sum',
        # Transform bound constraints
        'contrib.constraints_to_var_bounds',
        # Detect fixed variables
        'contrib.detect_fixed_vars',
        # Remove terms in equal to zero summations
        'contrib.propagate_zero_sum',
        # Remove trivial constraints
        'contrib.deactivate_trivial_constraints']
    for xfrm in preprocessing_transformations:
        TransformationFactory(xfrm).apply_to(m)

    # Deactivate extraneous IMPORT/EXPORT suffixes
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

    # Create solver, check availability
    if not SolverFactory(config.mip_solver).available():
        raise RuntimeError(
            "MIP solver %s is not available." % config.mip_solver)

    # Callback immediately before solving NLP subproblem
    config.call_before_master_solve(m, solve_data)

    # We use LoggingIntercept in order to suppress the stupid "Loading a
    # SolverResults object with a warning status" warning message.
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.mip_solver).solve(
            m, **config.mip_solver_args)
    terminate_cond = results.solver.termination_condition
    if terminate_cond is tc.infeasibleOrUnbounded:
        # Linear solvers will sometimes tell me that it's infeasible or
        # unbounded during presolve, but fails to distinguish. We need to
        # resolve with a solver option flag on.
        results, terminate_cond = distinguish_mip_infeasible_or_unbounded(
            m, config)

    # Build and return results object
    mip_result = MasterProblemResult()
    mip_result.feasible = True
    mip_result.var_values = list(v.value for v in GDPopt.working_var_list)
    mip_result.pyomo_results = results
    mip_result.disjunct_values = list(
        disj.indicator_var.value for disj in GDPopt.working_disjuncts_list)

    if terminate_cond is tc.optimal or terminate_cond is tc.locallyOptimal:
        pass
    elif terminate_cond is tc.infeasible:
        config.logger.info(
            'Linear GDP is infeasible. '
            'Problem may have no more feasible discrete configurations.')
        mip_result.feasible = False
    elif terminate_cond is tc.maxTimeLimit:
        # TODO check that status is actually ok and everything is feasible
        config.logger.info(
            'Unable to optimize linear GDP problem within time limit. '
            'Using current solver feasible solution.')
    elif (terminate_cond is tc.other and
          results.solution.status is SolutionStatus.feasible):
        # load the solution and suppress the warning message by setting
        # solver status to ok.
        config.logger.info(
            'Linear GDP solver reported feasible solution, '
            'but not guaranteed to be optimal.')
    else:
        raise ValueError(
            'GDPopt unable to handle linear GDP '
            'termination condition '
            'of %s. Solver message: %s' %
            (terminate_cond, results.solver.message))

    return mip_result


def distinguish_mip_infeasible_or_unbounded(m, config):
    """Distinguish between an infeasible or unbounded solution.

    Linear solvers will sometimes tell me that a problem is infeasible or
    unbounded during presolve, but not distinguish between the two cases. We
    address this by solving again with a solver option flag on.

    """
    tmp_args = deepcopy(config.mip_solver_args)
    # TODO This solver option is specific to Gurobi.
    tmp_args['options']['DualReductions'] = 0
    with SuppressInfeasibleWarning():
        results = SolverFactory(config.mip_solver).solve(m, **tmp_args)
    termination_condition = results.solver.termination_condition
    return results, termination_condition


def solve_LOA_master(solve_data, config):
    """Solve the augmented lagrangean outer approximation master problem."""
    m = solve_data.linear_GDP.clone()
    GDPopt = m.GDPopt_utils

    # Set up augmented Lagrangean penalty objective
    GDPopt.objective.deactivate()
    sign_adjust = 1 if GDPopt.objective.sense == minimize else -1
    GDPopt.OA_penalty_expr = Expression(
        expr=sign_adjust * config.OA_penalty_factor *
        sum(v for v in m.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
            if v.parent_component().local_name == 'GDPopt_OA_slacks'))
    GDPopt.oa_obj = Objective(
        expr=GDPopt.objective.expr + GDPopt.OA_penalty_expr,
        sense=GDPopt.objective.sense)
    solve_data.mip_iteration += 1

    mip_result = solve_linear_GDP(m, solve_data, config)
    if mip_result.feasible:
        if GDPopt.objective.sense == minimize:
            solve_data.LB = max(value(GDPopt.oa_obj.expr), solve_data.LB)
        else:
            solve_data.UB = min(value(GDPopt.oa_obj.expr), solve_data.UB)
        solve_data.iteration_log[
            (solve_data.master_iteration,
             solve_data.mip_iteration,
             solve_data.nlp_iteration)
        ] = (
            value(GDPopt.oa_obj.expr),
            value(GDPopt.objective.expr),
            mip_result.var_values
        )
        config.logger.info(
            'ITER %s.%s.%s-MIP: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.master_iteration,
               solve_data.mip_iteration,
               solve_data.nlp_iteration,
               value(GDPopt.oa_obj.expr),
               solve_data.LB, solve_data.UB))
    else:
        # Master problem was infeasible.
        if solve_data.master_iteration == 1:
            config.logger.warning(
                'GDPopt initialization may have generated poor '
                'quality cuts.')
        # set optimistic bound to infinity
        if GDPopt.objective.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
    # Call the MILP post-solve callback
    config.call_after_master_solve(m, solve_data)

    return mip_result


def solve_GLOA_master(solve_data, config):
    """Solve the rigorous outer approximation master problem."""
    m = solve_data.linear_GDP.clone()
    GDPopt = m.GDPopt_utils
    solve_data.mip_iteration += 1

    mip_result = solve_linear_GDP(m, solve_data, config)
    if mip_result.feasible:
        if GDPopt.objective.sense == minimize:
            solve_data.LB = max(value(GDPopt.objective.expr), solve_data.LB)
        else:
            solve_data.UB = min(value(GDPopt.objective.expr), solve_data.UB)
        solve_data.iteration_log[
            (solve_data.master_iteration,
             solve_data.mip_iteration,
             solve_data.nlp_iteration)
        ] = (
            value(GDPopt.objective.expr),
            value(GDPopt.objective.expr),
            mip_result.var_values
        )
        config.logger.info(
            'ITER %s.%s.%s-MIP: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.master_iteration,
               solve_data.mip_iteration,
               solve_data.nlp_iteration,
               value(GDPopt.objective.expr),
               solve_data.LB, solve_data.UB))
    else:
        # Master problem was infeasible.
        if solve_data.master_iteration == 1:
            config.logger.warning(
                'GDPopt initialization may have generated poor '
                'quality cuts.')
        # set optimistic bound to infinity
        if GDPopt.objective.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = float('-inf')
    # Call the MILP post-solve callback
    config.call_after_master_solve(m, solve_data)

    return mip_result
