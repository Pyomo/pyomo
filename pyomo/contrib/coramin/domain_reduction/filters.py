from pyomo.common.collections import ComponentSet
from coramin.domain_reduction.obbt import _bt_prep, _bt_cleanup
import pyomo.environ as pe
from pyomo.core.expr.numeric_expr import LinearExpression
import logging
from pyomo.contrib import appsi
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.block import _BlockData
from typing import Sequence, Optional, Union


logger = logging.getLogger(__name__)


def filter_variables_from_solution(candidate_variables_at_relaxation_solution, tolerance=1e-6):
    """
    This function takes a set of candidate variables for OBBT and filters out 
    the variables that are at their bounds in the provided solution to the 
    relaxation. See 

    Gleixner, Ambros M., et al. "Three enhancements for
    optimization-based bound tightening." Journal of Global
    Optimization 67.4 (2017): 731-757.

    for details on why this works. The basic idea is that if x = xl is
    feasible for the relaxation that will be used for OBBT, then
    minimizing x subject to that relaxation is guaranteed to result in
    an optimal solution of x* = xl.

    This function simply loops through
    candidate_variables_at_relaxation_solution and specifies which
    variables should be minimized and which variables should be
    maximized with OBBT.

    Parameters
    ----------
    candidate_variables_at_relaxation_solution: iterable of _GeneralVarData
        This should be an iterable of the variables which are candidates 
        for OBBT. The values of the variables should be feasible for the 
        relaxation that would be used to perform OBBT on the variables.
    tolerance: float
        A float greater than or equal to zero. If the value of the variable
        is within tolerance of its lower bound, then that variable is filtered
        from the set of variables that should be minimized for OBBT. The same
        is true for upper bounds and variables that should be maximized.

    Returns
    -------
    vars_to_minimize: ComponentSet of _GeneralVarData
        variables that should be considered for minimization
    vars_to_maximize: ComponentSet of _GeneralVarData
        variables that should be considered for maximization
    """
    candidate_vars = ComponentSet(candidate_variables_at_relaxation_solution)
    vars_to_minimize = ComponentSet()
    vars_to_maximize = ComponentSet()

    for v in candidate_vars:
        if (not v.has_lb()) or (v.value - v.lb > tolerance):
            vars_to_minimize.add(v)
        if (not v.has_ub()) or (v.ub - v.value > tolerance):
            vars_to_maximize.add(v)

    return vars_to_minimize, vars_to_maximize


def aggressive_filter(
    candidate_variables: Sequence[_GeneralVarData],
    relaxation: _BlockData,
    solver: Union[appsi.base.Solver, appsi.base.PersistentSolver],
    tolerance: float = 1e-6,
    objective_bound: Optional[float] = None,
    max_iter: int = 10,
    improvement_threshold: int = 5
):
    """
    This function takes a set of candidate variables for OBBT and filters out 
    the variables for which it does not make senese to perform OBBT on. See 

    Gleixner, Ambros M., et al. "Three enhancements for
    optimization-based bound tightening." Journal of Global
    Optimization 67.4 (2017): 731-757.

    for details. The basic idea is that if x = xl is
    feasible for the relaxation that will be used for OBBT, then
    minimizing x subject to that relaxation is guaranteed to result in
    an optimal solution of x* = xl.

    This function solves a series of optimization problems to try to 
    filter as many variables as possible.

    Parameters
    ----------
    candidate_variables: iterable of _GeneralVarData
        This should be an iterable of the variables which are candidates 
        for OBBT.
    relaxation: Block
        a convex relaxation
    solver: appsi.base.Solver
    tolerance: float
        A float greater than or equal to zero. If the value of the variable
        is within tolerance of its lower bound, then that variable is filtered
        from the set of variables that should be minimized for OBBT. The same
        is true for upper bounds and variables that should be maximized.
    objective_bound: float
        Primal bound for the objective
    max_iter: int
        Maximum number of iterations
    improvement_threshold: int
        If the number of filtered variables is less than improvement_threshold, then
        the filtering is terminated

    Returns
    -------
    vars_to_minimize: list of _GeneralVarData
        variables that should be considered for minimization
    vars_to_maximize: list of _GeneralVarData
        variables that should be considered for maximization
    """
    vars_to_minimize = ComponentSet(candidate_variables)
    vars_to_maximize = ComponentSet(candidate_variables)
    if len(candidate_variables) == 0:
        return vars_to_minimize, vars_to_maximize

    tmp = _bt_prep(model=relaxation, solver=solver, objective_bound=objective_bound)
    initial_var_values, deactivated_objectives, orig_update_config, orig_config = tmp

    vars_unbounded_from_below = ComponentSet()
    vars_unbounded_from_above = ComponentSet()
    for v in list(vars_to_minimize):
        if v.lb is None:
            vars_unbounded_from_below.add(v)
            vars_to_minimize.remove(v)
    for v in list(vars_to_maximize):
        if v.ub is None:
            vars_unbounded_from_above.add(v)
            vars_to_maximize.remove(v)

    for _set in [vars_to_minimize, vars_to_maximize]:
        for _iter in range(max_iter):
            if _set is vars_to_minimize:
                obj_coefs = [1 for v in _set]
            else:
                obj_coefs = [-1 for v in _set]
            obj_vars = list(_set)
            relaxation.__filter_obj = pe.Objective(expr=LinearExpression(linear_coefs=obj_coefs, linear_vars=obj_vars))
            if solver.is_persistent():
                solver.set_objective(relaxation.__filter_obj)
            solver.config.load_solution = False
            res = solver.solve(relaxation)
            if res.termination_condition == appsi.base.TerminationCondition.optimal:
                res.solution_loader.load_vars()
                success = True
            else:
                success = False
            del relaxation.__filter_obj

            if not success:
                break

            num_filtered = 0
            for v in list(_set):
                should_filter = False
                if _set is vars_to_minimize:
                    if v.value - v.lb <= tolerance:
                        should_filter = True
                else:
                    if v.ub - v.value <= tolerance:
                        should_filter = True
                if should_filter:
                    num_filtered += 1
                    _set.remove(v)
            logger.debug('filtered {0} vars on iter {1}'.format(num_filtered, _iter))

            if len(_set) == 0:
                break
            if num_filtered < improvement_threshold:
                break

    for v in vars_unbounded_from_below:
        vars_to_minimize.add(v)
    for v in vars_unbounded_from_above:
        vars_to_maximize.add(v)

    _bt_cleanup(
        model=relaxation, solver=solver, vardatalist=None,
        initial_var_values=initial_var_values,
        deactivated_objectives=deactivated_objectives,
        orig_update_config=orig_update_config, orig_config=orig_config,
        lower_bounds=None, upper_bounds=None
    )

    return vars_to_minimize, vars_to_maximize
