"""Provides functions to compute disjunctive variable bounds.

These are tighter variable bounds that are valid within the scope of a certain
disjunct.
"""
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.base.expr import identify_variables
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core import (Constraint, Objective,
                           TransformationFactory, maximize, minimize, value)
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.util.plugin import alias
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc
import textwrap
from six import iterkeys


def disjunctive_bound(var, scope):
    """Compute the disjunctive bounds for a variable in a given scope.

    Args:
        var (_VarData): Variable for which to compute bound
        scope (Component): The scope in which to compute the bound. If not a
            _DisjunctData, it will walk up the tree and use the scope of the
            most immediate enclosing _DisjunctData.

    Returns:
        numeric: the maximum of either the disjunctive lower bound, the
            variable lower bound, or None if neither exist.

    """
    # Initialize to the global variable bound
    var_bnd = (var.lb, var.ub)
    possible_disjunct = scope
    while possible_disjunct is not None:
        try:
            disj_bnd = possible_disjunct._disjunctive_bounds.get(
                var, (None, None))
            disj_bnd = (
                max(var_bnd[0], disj_bnd[0]) \
                    if disj_bnd[0] is not None else var_bnd[0],
                min(var_bnd[1], disj_bnd[1]) \
                    if disj_bnd[1] is not None else var_bnd[1]
            )
            return disj_bnd
        except AttributeError:
            # possible disjunct does not have attribute '_disjunctive_bounds'.
            # Try again with the scope's parent block.
            possible_disjunct = possible_disjunct.parent_block()
    # Unable to find '_disjunctive_bounds' attribute within search scope.
    return var_bnd


def disjunctive_lb(var, scope):
    """Compute the disjunctive lower bound for a variable in a given scope."""
    return disjunctive_bound(var, scope)[0]


def disjunctive_ub(var, scope):
    """Compute the disjunctive upper bound for a variable in a given scope."""
    return disjunctive_bound(var, scope)[1]


class ComputeDisjunctiveVarBounds(Transformation):
    """Compute disjunctive bounds in a given scope.

    Tries to compute the disjunctive bounds for all variables found in
    constraints that are in disjuncts under the given scope.

    This function uses the linear relaxation of the model to try to compute
    disjunctive bounds.

    Args:
        scope (Component): The scope under which to look for disjuncts.

    """

    alias('contrib.compute_disjunctive_bounds',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, scope):
        """Apply the transformation.

        Args:
            scope: Pyomo model object on which to compute disjuctive bounds.

        """
        disjuncts_to_process = list(scope.component_data_objects(
            ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
            descent_order=TraversalStrategy.BreadthFirstSearch))
        if scope.type() == Disjunct:
            disjuncts_to_process.insert(0, scope)

        for disjunct in disjuncts_to_process:
            # fix the disjunct to active, deactivate all nonlinear constraints,
            # and apply the big-M transformation
            old_disjunct_state = {'fixed': disjunct.indicator_var.fixed,
                                  'value': disjunct.indicator_var.value}

            disjunct.indicator_var.fix(1)
            scope._tmp_var_set = ComponentSet()
            # Maps a variable in a cloned model instance to the original model
            # variable
            for constraint in disjunct.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True):
                if constraint.body.polynomial_degree() in [0, 1]:
                    scope._tmp_var_set.update(
                        identify_variables(constraint.body))
            scope._var_list = list(scope._tmp_var_set)
            bigM_model = scope.clone()
            new_var_to_orig = ComponentMap(
                zip(bigM_model._var_list, scope._var_list))
            for constraint in bigM_model.component_data_objects(
                    ctype=Constraint, active=True,
                    descend_into=(Block, Disjunct)):
                if constraint.body.polynomial_degree() not in [0, 1]:
                    constraint.deactivate()

            TransformationFactory('gdp.bigm').apply_to(bigM_model)
            for var in bigM_model._tmp_var_set:
                # If variable is fixed, no need to calculate disjunctive bounds
                if var.fixed:
                    continue
                # calculate the disjunctive variable bounds for these variables
                # disable all other objectives
                for obj in bigM_model.component_data_objects(ctype=Objective):
                    obj.deactivate()
                bigM_model.del_component('_var_bounding_obj')
                # Calculate the lower bound
                bigM_model._var_bounding_obj = Objective(
                    expr=var, sense=minimize)
                results = SolverFactory('cbc').solve(bigM_model)
                if results.solver.termination_condition is tc.optimal:
                    disj_lb = value(var)
                elif results.solver.termination_condition is tc.infeasible:
                    disj_lb = None
                else:
                    raise NotImplementedError(
                        "Unhandled termination condition: %s"
                        % results.solver.termination_condition)
                # Calculate the upper bound
                bigM_model._var_bounding_obj.sense = maximize
                results = SolverFactory('cbc').solve(bigM_model)
                if results.solver.termination_condition is tc.optimal:
                    disj_ub = value(var)
                elif results.solver.termination_condition is tc.infeasible:
                    disj_ub = None
                else:
                    raise NotImplementedError(
                        "Unhandled termination condition: %s"
                        % results.solver.termination_condition)
                disjunct._disjunctive_bounds[new_var_to_orig[var]] = \
                    (disj_lb, disj_ub)

            # reset the disjunct
            if not old_disjunct_state['fixed']:
                disjunct.indicator_var.unfix()
            disjunct.indicator_var.set_value(old_disjunct_state['value'])
