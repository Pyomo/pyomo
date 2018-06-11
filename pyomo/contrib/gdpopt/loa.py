"""Main functions for Logic-based outer approximation (LOA)."""
from __future__ import division

from pyomo.contrib.gdpopt.mip_solve import solve_linear_GDP
from pyomo.core.base import Block, Expression, Objective, Var, minimize, value
from pyomo.gdp import Disjunct


def solve_OA_master(solve_data, config):
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

    mip_results = solve_linear_GDP(m, solve_data, config)
    if mip_results:
        if GDPopt.objective.sense == minimize:
            solve_data.LB = max(value(GDPopt.oa_obj.expr), solve_data.LB)
            solve_data.LB_progress.append(solve_data.LB)
        else:
            solve_data.UB = min(value(GDPopt.oa_obj.expr), solve_data.UB)
            solve_data.UB_progress.append(solve_data.UB)
        config.logger.info(
            'ITER %s.%s-MIP: OBJ: %s  LB: %s  UB: %s'
            % (solve_data.master_iteration,
               solve_data.subproblem_iteration,
               value(GDPopt.oa_obj.expr),
               solve_data.LB, solve_data.UB))
    else:
        if solve_data.master_iteration == 1:
            config.logger.warning(
                'GDPopt initialization may have generated poor '
                'quality cuts.')
        # set optimistic bound to infinity
        if GDPopt.objective.sense == minimize:
            solve_data.LB = float('inf')
            solve_data.LB_progress.append(solve_data.UB)
        else:
            solve_data.UB = float('-inf')
            solve_data.UB_progress.append(solve_data.UB)
    # Call the MILP post-solve callback
    config.master_postsolve(m, solve_data)

    return mip_results
