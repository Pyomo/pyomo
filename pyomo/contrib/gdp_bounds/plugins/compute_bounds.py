"""Provides functions to compute and enforce disjunctive variable bounds.

These are tighter variable bounds that are valid within the scope of a certain
disjunct. That is, these are bounds on variables that hold true when an
associated disjunct is active (indicator_var value is 1). The bounds are
enforced using a ConstraintList on every Disjunct, enforcing the relevant
variable bounds. This may lead to duplication of constraints, so the constraints
to variable bounds preprocessing transformation is recommended for NLP problems
processed with this transformation.

"""
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr.current import identify_variables
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core import (Constraint, Objective, ConstraintList,
                           TransformationFactory, maximize, minimize, value)
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc
import textwrap
from six import iterkeys, iteritems
linear_degrees = set([0, 1])


def disjunctive_bound(var, scope):
    """Compute the disjunctive bounds for a variable in a given scope.

    Args:
        var (_VarData): Variable for which to compute bound
        scope (Component): The scope in which to compute the bound. If not a
            _DisjunctData, it will walk up the tree and use the scope of the
            most immediate enclosing _DisjunctData.

    Returns:
        numeric: the tighter of either the disjunctive lower bound, the
            variable lower bound, or None if neither exist.

    """
    # Initialize to the global variable bound
    var_bnd = (value(var.lb, exception=False), value(var.ub, exception=False))
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


@TransformationFactory.register('contrib.compute_disj_var_bounds',
          doc="Compute disjunctive bounds in a given model.")
class ComputeDisjunctiveVarBounds(Transformation):
    """Compute disjunctive bounds in a given model.

    Tries to compute the disjunctive bounds for all variables found in
    constraints that are in disjuncts under the given model.

    This function uses the linear relaxation of the model to try to compute
    disjunctive bounds.

    Args:
        model (Component): The model under which to look for disjuncts.

    """

    def _apply_to(self, model):
        """Apply the transformation.

        Args:
            model: Pyomo model object on which to compute disjuctive bounds.

        """
        disjuncts_to_process = list(model.component_data_objects(
            ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
            descent_order=TraversalStrategy.BreadthFirstSearch))
        if model.type() == Disjunct:
            disjuncts_to_process.insert(0, model)

        # Deactivate nonlinear constraints
        model._tmp_constr_deactivated = ComponentSet()
        for constraint in model.component_data_objects(
                ctype=Constraint, active=True,
                descend_into=(Block, Disjunct)):
            if constraint.body.polynomial_degree() not in linear_degrees:
                model._tmp_constr_deactivated.add(constraint)
                constraint.deactivate()

        for disjunct in disjuncts_to_process:
            # If disjunct does not have a component map to store disjunctive
            # bounds, then make one.
            if not hasattr(disjunct, '_disj_var_bounds'):
                disjunct._disj_var_bounds = ComponentMap()

            # fix the disjunct to active, deactivate all nonlinear constraints,
            # and apply the big-M transformation
            old_disjunct_state = {'fixed': disjunct.indicator_var.fixed,
                                  'value': disjunct.indicator_var.value}

            disjunct.indicator_var.fix(1)
            model._tmp_var_set = ComponentSet()
            # Maps a variable in a cloned model instance to the original model
            # variable
            for constraint in disjunct.component_data_objects(
                    ctype=Constraint, active=True, descend_into=True):
                model._tmp_var_set.update(identify_variables(constraint.body))
            model._var_list = list(model._tmp_var_set)
            bigM_model = model.clone()
            new_var_to_orig = ComponentMap(
                zip(bigM_model._var_list, model._var_list))

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
                    # TODO disjunct can be fathomed?
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
                    # TODO disjunct can be fathomed?
                else:
                    raise NotImplementedError(
                        "Unhandled termination condition: %s"
                        % results.solver.termination_condition)
                old_bounds = disjunct._disj_var_bounds.get(
                    new_var_to_orig[var], (None, None)  # default of None
                )
                # update bounds values
                disjunct._disj_var_bounds[new_var_to_orig[var]] = (
                    min_if_not_None(disj_lb, old_bounds[0]),
                    max_if_not_None(disj_ub, old_bounds[1]))

            # reset the disjunct
            if not old_disjunct_state['fixed']:
                disjunct.indicator_var.unfix()
            disjunct.indicator_var.set_value(old_disjunct_state['value'])

            # Enforce the disjunctive variable bounds as constraints
            if hasattr(disjunct, '_disjunctive_var_constraints'):
                if getattr(disjunct._disjunctive_var_constraints, 'doc', "")\
                        .startswith("q.Autogenerated"):
                    del disjunct._disjunctive_var_constraints
                else:
                    raise ValueError(
                        'Disjunct %s already has an attribute '
                        '_disjunctive_var_constraints required by the '
                        'gdp_bounds package.')
            cons_list = disjunct._disjunctive_var_constraints = ConstraintList(
                doc="q.Autogenerated constraints enforcing "
                "disjunctive variable bounds."
            )
            for var, bounds in iteritems(disjunct._disj_var_bounds):
                lbb, ubb = bounds
                if lbb is not None:
                    cons_list.add(expr=lbb <= var)
                if ubb is not None:
                    cons_list.add(expr=var <= ubb)

        # Reactivate deactivated nonlinear constraints
        for constraint in model._tmp_constr_deactivated:
            constraint.activate()


def min_if_not_None(*args):
    """Returns the minimum among non-None elements.

    Returns None is no non-None elements exist.

    """
    non_nones = [a for a in args if a is not None]
    return min(non_nones or [None])  # handling for empty non_nones list


def max_if_not_None(*args):
    """Returns the maximum among non-None elements.

    Returns None is no non-None elements exist.

    """
    non_nones = [a for a in args if a is not None]
    return max(non_nones or [None])  # handling for empty non_nones list
