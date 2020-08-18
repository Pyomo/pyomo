"""Provides functions for retrieving disjunctive variable bound information stored on a model."""
from pyomo.common.collections import ComponentMap
from pyomo.core import value

inf = float('inf')


def disjunctive_bounds(scope):
    """Return all of the variable bounds defined at a disjunctive scope."""
    possible_disjunct = scope
    while possible_disjunct is not None:
        try:
            return possible_disjunct._disj_var_bounds
        except AttributeError:
            # possible disjunct does not have attribute '_disj_var_bounds'.
            # Try again with the scope's parent block.
            possible_disjunct = possible_disjunct.parent_block()
    # Unable to find '_disj_var_bounds' attribute within search scope.
    return ComponentMap()


def disjunctive_bound(var, scope):
    """Compute the disjunctive bounds for a variable in a given scope.

    Args:
        var (_VarData): Variable for which to compute bound
        scope (Component): The scope in which to compute the bound. If not a
            _DisjunctData, it will walk up the tree and use the scope of the
            most immediate enclosing _DisjunctData.

    Returns:
        numeric: the tighter of either the disjunctive lower bound, the
            variable lower bound, or (-inf, inf) if neither exist.

    """
    # Initialize to the global variable bound
    var_bnd = (
        value(var.lb) if var.has_lb() else -inf,
        value(var.ub) if var.has_ub() else inf)
    possible_disjunct = scope
    while possible_disjunct is not None:
        try:
            disj_bnd = possible_disjunct._disj_var_bounds.get(var, (-inf, inf))
            disj_bnd = (
                max(var_bnd[0], disj_bnd[0]),
                min(var_bnd[1], disj_bnd[1]))
            return disj_bnd
        except AttributeError:
            # possible disjunct does not have attribute '_disj_var_bounds'.
            # Try again with the scope's parent block.
            possible_disjunct = possible_disjunct.parent_block()
    # Unable to find '_disj_var_bounds' attribute within search scope.
    return var_bnd


def disjunctive_lb(var, scope):
    """Compute the disjunctive lower bound for a variable in a given scope."""
    return disjunctive_bound(var, scope)[0]


def disjunctive_ub(var, scope):
    """Compute the disjunctive upper bound for a variable in a given scope."""
    return disjunctive_bound(var, scope)[1]
