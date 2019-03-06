"""Provides functions to compute and enforce disjunctive variable bounds.

These are tighter variable bounds that are valid within the scope of a certain
disjunct. That is, these are bounds on variables that hold true when an
associated disjunct is active (indicator_var value is 1). The bounds are
enforced using a ConstraintList on every Disjunct, enforcing the relevant
variable bounds. This may lead to duplication of constraints, so the constraints
to variable bounds preprocessing transformation is recommended for NLP problems
processed with this transformation.

"""
from pyomo.contrib.fbbt.fbbt import fbbt_block
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr.current import identify_variables
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core import (Constraint, Objective,
                        TransformationFactory, maximize, minimize, value)
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc

linear_degrees = {0, 1}
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
            disj_bnd = possible_disjunct._disj_var_bounds.get(
                var, (None, None))
            disj_bnd = (
                max(var_bnd[0], disj_bnd[0]) \
                    if disj_bnd[0] is not None else var_bnd[0],
                min(var_bnd[1], disj_bnd[1]) \
                    if disj_bnd[1] is not None else var_bnd[1]
            )
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


def disjunctive_obbt(model, solver):
    """Provides Optimality-based bounds tightening to a model using a solver."""
    model._disjuncts_to_process = list(model.component_data_objects(
        ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
        descent_order=TraversalStrategy.BreadthFirstSearch))
    if model.type() == Disjunct:
        model._disjuncts_to_process.insert(0, model)

    linear_var_set = ComponentSet()
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() in linear_degrees:
            linear_var_set.update(identify_variables(constr.body, include_fixed=False))
    model._disj_bnds_linear_vars = list(linear_var_set)

    for disj_idx, disjunct in enumerate(model._disjuncts_to_process):
        var_bnds = obbt_disjunct(model, disj_idx, solver)
        if var_bnds is not None:
            # Add bounds to the disjunct
            if not hasattr(disjunct, '_disj_var_bounds'):
                # No bounds had been computed before. Attach the bounds dictionary.
                disjunct._disj_var_bounds = var_bnds
            else:
                # Update the bounds dictionary.
                for var, new_bnds in var_bnds.items():
                    old_lb, old_ub = disjunct._disj_var_bounds.get(var, (-inf, inf))
                    new_lb, new_ub = new_bnds
                    disjunct._disj_var_bounds[var] = (max(old_lb, new_lb), min(old_ub, new_ub))
        else:
            disjunct.deactivate()  # prune disjunct


def obbt_disjunct(orig_model, idx, solver):
    model = orig_model.clone()

    # Fix the disjunct to be active
    disjunct = model._disjuncts_to_process[idx]
    disjunct.indicator_var.fix(1)

    for obj in model.component_data_objects(Objective, active=True):
        obj.deactivate()

    # Deactivate nonlinear constraints
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() not in linear_degrees:
            constr.deactivate()

    # Only look at the variables participating in active constraints within the scope
    relevant_var_set = ComponentSet()
    for constr in disjunct.component_data_objects(Constraint, active=True):
        relevant_var_set.update(identify_variables(constr.body, include_fixed=False))

    TransformationFactory('gdp.bigm').apply_to(model)

    for var in relevant_var_set:
        model._var_bounding_obj = Objective(expr=var, sense=minimize)
        var_lb = solve_bounding_problem(model, solver)
        if var_lb is None:
            return None  # bounding problem infeasible
        model._var_bounding_obj.set_value(expr=-var)
        var_ub = solve_bounding_problem(model, solver)
        if var_ub is None:
            return None  # bounding problem infeasible
        else:
            var_ub = -var_ub  # sign correction

        var.setlb(var_lb)
        var.setub(var_ub)

    # Maps original variable --> (new computed LB, new computed UB)
    var_bnds = ComponentMap(
        ((orig_var, (
            clone_var.lb if clone_var.has_lb() else -inf,
            clone_var.ub if clone_var.has_ub() else inf))
         for orig_var, clone_var in zip(
            orig_model._disj_bnds_linear_vars, model._disj_bnds_linear_vars)
         if clone_var in relevant_var_set)
    )
    return var_bnds


def solve_bounding_problem(model, solver):
    results = SolverFactory(solver).solve(model)
    if results.solver.termination_condition is tc.optimal:
        return value(model._var_bounding_obj.expr)
    elif results.solver.termination_condition is tc.infeasible:
        return None
    elif results.solver.termination_condition is tc.unbounded:
        return -inf
    else:
        raise NotImplementedError(
            "Unhandled termination condition: %s"
            % results.solver.termination_condition)


def disjunctive_fbbt(model):
    """Applies FBBT to a model"""
    fbbt_disjunct(model, ComponentMap())


def fbbt_disjunct(disj, parent_bounds):
    orig_bnds = ComponentMap(parent_bounds)
    try:
        for var, var_bnds in disj._disj_var_bounds.items():
            scope_lb, scope_ub = var_bnds
            parent_lb, parent_ub = parent_bounds.get(var, (-inf, inf))
            orig_bnds[var] = (max(scope_lb, parent_lb), min(scope_ub, parent_ub))
    except AttributeError:
        # disj._disj_var_bounds does not exist yet
        pass
    new_bnds = fbbt_block(disj, update_variable_bounds=False, initial_bounds=orig_bnds)
    disj._disj_var_bounds = new_bnds
    # Handle nested disjuncts
    for disj in disj.component_data_objects(Disjunct, active=True):
        fbbt_disjunct(disj, new_bnds)


@TransformationFactory.register('contrib.compute_disj_var_bounds',
          doc="Compute disjunctive bounds in a given model.")
class ComputeDisjunctiveVarBounds(Transformation):
    """Compute disjunctive bounds in a given model.

    Tries to compute the disjunctive bounds for all variables found in
    constraints that are in disjuncts under the given model.

    Two strategies are available to compute the disjunctive bounds:
     - Feasibility-based bounds tightening using the contrib.fbbt package. (Default)
     - Optimality-based bounds tightening by solving the linear relaxation of the model.

    This transformation introduces ComponentMap objects named _disj_var_bounds to
    each Disjunct and the top-level model object. These map var --> (var.disj_lb, var.disj_ub)
    for each disjunctive scope.

    Args:
        model (Component): The model under which to look for disjuncts.
        solver (string): The solver to use for OBBT, or None for FBBT.

    """

    def _apply_to(self, model, solver=None):
        """Apply the transformation.

        Args:
            model: Pyomo model object on which to compute disjuctive bounds.

        """
        if solver is not None:
            disjunctive_obbt(model, solver)
        else:
            disjunctive_fbbt(model)
