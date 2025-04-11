#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Transformation to aggregate equal variables."""


from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base import (
    Block,
    Constraint,
    VarList,
    Objective,
    Reals,
    TransformationFactory,
)
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
import logging

logger = logging.getLogger('pyomo.contrib.preprocessing')


def _get_equality_linked_variables(constraint):
    """Return the two variables linked by an equality constraint x == y.

    If the constraint does not match this form, skip it.

    """
    if value(constraint.lower) != 0 or value(constraint.upper) != 0:
        # LB and UB on constraint must be zero; otherwise, return empty tuple.
        return ()
    if constraint.body.polynomial_degree() != 1:
        # must be a linear constraint; otherwise, return empty tuple.
        return ()

    # Generate the standard linear representation
    repn = generate_standard_repn(constraint.body)
    nonzero_coef_vars = tuple(
        v
        for i, v in enumerate(repn.linear_vars)
        # if coefficient on variable is nonzero
        if repn.linear_coefs[i] != 0
    )
    if len(nonzero_coef_vars) != 2:
        # Expect two variables with nonzero coefficient in constraint;
        # otherwise, return empty tuple.
        return ()
    if sorted(coef for coef in repn.linear_coefs if coef != 0) != [-1, 1]:
        # Expect a constraint of form x == y --> 0 == -1 * x + 1 * y;
        # otherwise, return empty tuple.
        return ()
    # Above checks are satisfied. Return the variables.
    return nonzero_coef_vars


def _fix_equality_fixed_variables(model, scaling_tolerance=1e-10):
    """Detects variables fixed by a constraint: ax=b.

    Fixes the variable to the constant value (b/a) and deactivates the relevant
    constraint.

    This sub-transformation is different than contrib.detect_fixed_vars because
    it looks for x = const rather than x.lb = x.ub.

    """
    for constraint in model.component_data_objects(
        ctype=Constraint, active=True, descend_into=True
    ):
        if not (constraint.has_lb() and constraint.has_ub()):
            # Constraint is not an equality. Skip.
            continue
        if value(constraint.lower) != value(constraint.upper):
            # Constraint is not an equality. Skip.
            continue
        if constraint.body.polynomial_degree() != 1:
            # Constraint is not linear. Skip.
            continue

        # Generate the standard linear representation
        repn = generate_standard_repn(constraint.body)
        # Generator of tuples with the coefficient and variable object for
        # nonzero coefficients.
        nonzero_coef_vars = (
            (repn.linear_coefs[i], v)
            for i, v in enumerate(repn.linear_vars)
            # if coefficient on variable is nonzero
            if repn.linear_coefs[i] != 0
        )
        # get the coefficient and variable object
        coef, var = next(nonzero_coef_vars)
        if next(nonzero_coef_vars, None) is not None:
            # Expect one variable with nonzero coefficient in constraint;
            # otherwise, skip.
            continue
        # Constant term on the constraint body
        const = repn.constant if repn.constant is not None else 0

        if abs(coef) <= scaling_tolerance:
            logger.warning(
                "Skipping fixed variable processing for constraint %s: "
                "%s * %s + %s = %s because coefficient %s is below "
                "tolerance of %s. Check your problem scaling."
                % (
                    constraint.name,
                    coef,
                    var.name,
                    const,
                    value(constraint.lower),
                    coef,
                    scaling_tolerance,
                )
            )
            continue

        # Constraint has form lower <= coef * var + const <= upper. We know that
        # lower = upper, so coef * var + const = lower.
        var_value = (value(constraint.lower) - const) / coef

        var.fix(var_value)
        constraint.deactivate()


def _build_equality_set(model):
    """Construct an equality set map.

    Maps all variables to the set of variables that are linked to them by
    equality. Mapping takes place using id(). That is, if you have x = y, then
    you would have id(x) -> ComponentSet([x, y]) and id(y) -> ComponentSet([x,
    y]) in the mapping.

    """
    # Map of variables to their equality set (ComponentSet)
    eq_var_map = ComponentMap()

    # Loop through all the active constraints in the model
    for constraint in model.component_data_objects(
        ctype=Constraint, active=True, descend_into=True
    ):
        eq_linked_vars = _get_equality_linked_variables(constraint)
        if not eq_linked_vars:
            continue  # if we get an empty tuple, skip to next constraint.
        v1, v2 = eq_linked_vars
        set1 = eq_var_map.get(v1, ComponentSet((v1, v2)))
        set2 = eq_var_map.get(v2, (v2,))

        # if set1 and set2 are equivalent, skip to next constraint.
        if set1 is set2:
            continue

        # add all elements of set2 to set 1
        set1.update(set2)
        # Update all elements to point to set 1
        for v in set1:
            eq_var_map[v] = set1

    return eq_var_map


# TODO: these two functions were copied from contrib.gdp_bounds.compute_bounds
# at some point, these all should be moved into pyomo.common.
def min_if_not_None(iterable):
    """Returns the minimum among non-None elements.

    Returns None when all elements are None.

    """
    non_nones = [a for a in iterable if a is not None]
    return min(non_nones or [None])  # min( [] or [None] ) -> None


def max_if_not_None(iterable):
    """Returns the maximum among non-None elements.

    Returns None when all elements are None.

    """
    non_nones = [a for a in iterable if a is not None]
    return max(non_nones or [None])  # min( [] or [None] ) -> None


@TransformationFactory.register(
    'contrib.aggregate_vars',
    doc="Aggregate model variables that are linked by equality constraints.",
)
class VariableAggregator(IsomorphicTransformation):
    r"""Aggregate model variables that are linked by equality constraints.

    Before:

    .. math::
        x = y \\
        a = 2x + 6y + 7 \\
        b = 5y + 6 \\

    After:

    .. math::
        z = x = y \\
        a = 8z + 7 \\
        b = 5z + 6

    .. warning::

       TODO: unclear what happens to "capital-E" Expressions at this
       point in time.

    """

    def _apply_to(self, model, detect_fixed_vars=True):
        """Apply the transformation to the given model."""
        # Generate the equality sets
        eq_var_map = _build_equality_set(model)

        # Detect and process fixed variables.
        if detect_fixed_vars:
            _fix_equality_fixed_variables(model)

        # Generate aggregation infrastructure
        model._var_aggregator_info = Block(
            doc="Holds information for the variable aggregation "
            "transformation system."
        )
        z = model._var_aggregator_info.z = VarList(doc="Aggregated variables.")
        # Map of the aggregate var to the equalty set (ComponentSet)
        z_to_vars = model._var_aggregator_info.z_to_vars = ComponentMap()
        # Map of variables to their corresponding aggregate var
        var_to_z = model._var_aggregator_info.var_to_z = ComponentMap()
        processed_vars = ComponentSet()

        # TODO This iteritems is sorted by the variable name of the key in
        # order to preserve determinism. Unfortunately, var.name() is an
        # expensive operation right now.
        for var, eq_set in sorted(eq_var_map.items(), key=lambda tup: tup[0].name):
            if var in processed_vars:
                continue  # Skip already-process variables

            # This would be weird. The variable hasn't been processed, but is
            # in the map. Raise an exception.
            assert var_to_z.get(var, None) is None

            z_agg = z.add()
            z_to_vars[z_agg] = eq_set
            var_to_z.update(ComponentMap((v, z_agg) for v in eq_set))

            # Set the bounds of the aggregate variable based on the bounds of
            # the variables in its equality set.
            z_agg.setlb(max_if_not_None(v.lb for v in eq_set if v.has_lb()))
            z_agg.setub(min_if_not_None(v.ub for v in eq_set if v.has_ub()))
            # Set the domain of the aggregate variable to the intersection of
            # the domains of the variables in its equality set
            domain = Reals
            for v in eq_set:
                domain = domain & v.domain
            z_agg.domain = domain

            # Set the fixed status of the aggregate var
            fixed_vars = [v for v in eq_set if v.fixed]
            if fixed_vars:
                # Check to make sure all the fixed values are the same.
                if any(var.value != fixed_vars[0].value for var in fixed_vars[1:]):
                    raise ValueError(
                        "Aggregate variable for equality set is fixed to "
                        "multiple different values: %s" % (fixed_vars,)
                    )
                z_agg.fix(fixed_vars[0].value)

                # Check that the fixed value lies within bounds.
                if z_agg.has_lb() and z_agg.value < value(z_agg.lb):
                    raise ValueError(
                        "Aggregate variable for equality set is fixed to "
                        "a value less than its lower bound: %s < LB %s"
                        % (z_agg.value, value(z_agg.lb))
                    )
                if z_agg.has_ub() and z_agg.value > value(z_agg.ub):
                    raise ValueError(
                        "Aggregate variable for equality set is fixed to "
                        "a value greater than its upper bound: %s > UB %s"
                        % (z_agg.value, value(z_agg.ub))
                    )
            else:
                # Set the value to be the average of the values within the
                # bounds only if the value is not already fixed.
                values_within_bounds = [
                    v.value
                    for v in eq_set
                    if (
                        v.value is not None
                        and (not z_agg.has_lb() or v.value >= value(z_agg.lb))
                        and (not z_agg.has_ub() or v.value <= value(z_agg.ub))
                    )
                ]
                if values_within_bounds:
                    z_agg.set_value(
                        sum(values_within_bounds) / len(values_within_bounds),
                        skip_validation=True,
                    )

            processed_vars.update(eq_set)

        # Do the substitution
        substitution_map = {id(var): z_var for var, z_var in var_to_z.items()}
        visitor = ExpressionReplacementVisitor(
            substitute=substitution_map,
            descend_into_named_expressions=True,
            remove_named_expressions=False,
        )
        for constr in model.component_data_objects(ctype=Constraint, active=True):
            orig_body = constr.body
            new_body = visitor.walk_expression(constr.body)
            if orig_body is not new_body:
                constr.set_value((constr.lower, new_body, constr.upper))

        for objective in model.component_data_objects(ctype=Objective, active=True):
            orig_expr = objective.expr
            new_expr = visitor.walk_expression(objective.expr)
            if orig_expr is not new_expr:
                objective.set_value(new_expr)

    def update_variables(self, model):
        """Update the values of the variables that were replaced by aggregates.

        TODO: reduced costs

        """
        datablock = model._var_aggregator_info
        for agg_var in datablock.z.itervalues():
            if not agg_var.stale:
                for var in datablock.z_to_vars[agg_var]:
                    # We don't want to accidentally trigger the reset of
                    # the global stale indicator, so we will set this
                    # variable to be "stale", knowing that set_value
                    # will switch it back to "not stale".  In normal
                    # situations, we would expect var to already be
                    # stale.
                    var.stale = True
                    var.set_value(agg_var.value, skip_validation=True)
