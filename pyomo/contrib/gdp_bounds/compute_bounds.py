#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Provides functions to compute and enforce disjunctive variable bounds.

These are tighter variable bounds that are valid within the scope of a certain
disjunct. That is, these are bounds on variables that hold true when an
associated disjunct is active (indicator_var value is 1). The bounds are
enforced using a ConstraintList on every Disjunct, enforcing the relevant
variable bounds. This may lead to duplication of constraints, so the constraints
to variable bounds preprocessing transformation is recommended for NLP problems
processed with this transformation.

"""
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt, BoundsManager
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr.current import identify_variables
from pyomo.core import (Constraint, Objective,
                        TransformationFactory, minimize, value)
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc

linear_degrees = {0, 1}
inf = float('inf')

def disjunctive_obbt(model, solver):
    """Provides Optimality-based bounds tightening to a model using a solver."""
    model._disjuncts_to_process = list(model.component_data_objects(
        ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
        descent_order=TraversalStrategy.BreadthFirstSearch))
    if model.ctype == Disjunct:
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

    model._var_bounding_obj = Objective(expr=1, sense=minimize)

    for var in relevant_var_set:
        model._var_bounding_obj.set_value(expr=var)
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
            scope_lb = -inf if scope_lb is None else scope_lb
            scope_ub = inf if scope_ub is None else scope_ub
            parent_lb, parent_ub = parent_bounds.get(var, (-inf, inf))
            orig_bnds[var] = (max(scope_lb, parent_lb), min(scope_ub, parent_ub))
    except AttributeError:
        # disj._disj_var_bounds does not exist yet
        pass
    bnds_manager = BoundsManager(disj)
    bnds_manager.load_bounds(orig_bnds)
    try:
        new_bnds = fbbt(disj)
    except InfeasibleConstraintException as e:
        if disj.ctype == Disjunct:
            disj.deactivate()  # simply prune the disjunct
        new_bnds = parent_bounds
    bnds_manager.pop_bounds()
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
