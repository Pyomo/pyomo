"""Provides functions to compute disjunctive variable bounds.

These are tighter variable bounds that are valid within the scope of a certain
disjunct.
"""
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr import identify_variables
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.environ import (Constraint, Objective, SolverFactory,
                           TransformationFactory, maximize, minimize, value)
from pyomo.gdp.disjunct import Disjunct


def disjunctive_lb(var, scope):
    """Compute the disjunctive lower bound for a variable in a given scope.

    Args:
        var (_VarData): Variable for which to compute bound
        scope (Component): The scope in which to compute the bound. If not a
            _DisjunctData, it will walk up the tree and use the scope of the
            most immediate enclosing _DisjunctData.

    Returns:
        numeric: the maximum of either the disjunctive lower bound, the
            variable lower bound, or None if neither exist.

    """
    disj_lb = var.lb
    possible_disjunct = scope
    while possible_disjunct is not None:
        try:
            disj_bnd = possible_disjunct._disjunctive_bounds.get(
                var, (None, None))[0]
            disj_lb = max(disj_lb, disj_bnd) \
                if disj_bnd is not None else disj_lb
        except AttributeError:
            pass
        possible_disjunct = possible_disjunct.parent_block()
    return disj_lb


def disjunctive_ub(var, scope):
    """Compute the disjunctive upper bound for a variable in a given scope.

    Args:
        var (_VarData): Variable for which to compute bound
        scope (Component): The scope in which to compute the bound. If not a
            _DisjunctData, it will walk up the tree and use the scope of the
            most immediate enclosing _DisjunctData.

    Returns:
        numeric: the minimum of either the disjunctive upper bound, the
            variable upper bound, or None if neither exist.

    """
    disj_ub = var.ub
    possible_disjunct = scope
    while possible_disjunct is not None:
        try:
            disj_bnd = possible_disjunct._disjunctive_bounds.get(
                var, (None, None))[1]
            disj_ub = min(disj_ub, disj_bnd) \
                if disj_bnd is not None else disj_ub
        except AttributeError:
            pass
        possible_disjunct = possible_disjunct.parent_block()
    return disj_ub


def compute_disjunctive_bounds(scope):
    """Compute disjunctive bounds in a given scope.

    Tries to compute the disjunctive bounds for all variables found in
    constraints that are in disjuncts under the given scope.

    This function uses the linear relaxation of the model to try to compute
    disjunctive bounds.

    Args:
        scope (Component): The scope under which to look for disjuncts.

    """
    disjuncts_to_process = [scope.component_data_objects(
        ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
        descent_strategy=TraversalStrategy.BreadthFirstSearch)]
    if scope.type() == Disjunct:
        disjuncts_to_process.insert(0, scope)

    for disjunct in disjuncts_to_process:
        # make sure the disjunct has the _disjunctive_bounds ComponentMap
        if not hasattr(disjunct, '_disjunctive_bounds'):
            disjunct._disjunctive_bounds = ComponentMap()

        # fix the disjunct to active, deactivate all nonlinear constraints, and
        # apply the big-M transformation
        old_disjunct_state = {'fixed': disjunct.indicator_var.fixed,
                              'value': value(disjunct.indicator_var)}

        disjunct.indicator_var.fix(1)
        var_set = ComponentSet()
        for constraint in disjunct.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct)):
            if constraint.body.polynomial_degree() not in [0, 1]:
                constraint.deactivate()
            else:
                var_set.update(identify_variables(constraint.body))

        bigM_model = TransformationFactory('gdp.bigm').create_using(
            disjunct)
        for var in var_set:
            # calculate the disjunctive variable bounds for these variables
            # disable all other objectives
            for obj in bigM_model.component_data_objects(ctype=Objective):
                obj.deactivate()
            bigM_model._var_bounding_obj = Objective(expr=var, sense=minimize)
            SolverFactory('gurobi').solve(bigM_model)
            # TODO check if solve successful
            disj_lb = value(var)
            bigM_model._var_bounding_obj.sense = maximize
            SolverFactory('gurobi').solve(bigM_model)
            # TODO check if solve successful
            disj_ub = value(var)
            disjunct._disjunctive_bounds[var] = (disj_lb, disj_ub)

        # reset the disjunct
        if not old_disjunct_state['fixed']:
            disjunct.indicator_var.unfix()
        disjunct.indicator_var.set_value(old_disjunct_state['value'])
