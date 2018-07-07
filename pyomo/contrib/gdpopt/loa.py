"""Main functions for Logic-based outer approximation (LOA)."""
from __future__ import division

from pyomo.contrib.gdpopt.mip_solve import solve_linear_GDP
from pyomo.contrib.gdpopt.nlp_solve import (solve_NLP,
                                            update_nlp_progress_indicators)
from pyomo.contrib.gdpopt.util import copy_and_fix_mip_values_to_nlp
from pyomo.core import (Block, Expression, Objective, TransformationFactory,
                        Var, minimize, value)
from pyomo.gdp import Disjunct


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


def solve_LOA_subproblem(mip_var_values, solve_data, config):
    """Set up and solve the local LOA subproblem."""
    nlp_model = solve_data.working_model.clone()
    solve_data.nlp_iteration += 1
    # copy in the discrete variable values
    copy_and_fix_mip_values_to_nlp(nlp_model.GDPopt_utils.working_var_list,
                                   mip_var_values, config)
    TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)

    nlp_result = solve_NLP(nlp_model, solve_data, config)
    if nlp_result.feasible:  # NLP is feasible
        update_nlp_progress_indicators(nlp_model, solve_data, config)
    return nlp_result
