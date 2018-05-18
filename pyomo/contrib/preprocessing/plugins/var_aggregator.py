"""Transformation to aggregate equal variables."""

from __future__ import division

import textwrap

from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
from pyomo.util.plugin import alias


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
    nonzero_coef_vars = tuple(v for i, v in enumerate(repn.linear_vars)
                         # if coefficient on variable is nonzero
                         if repn.linear_coefs[i] != 0)
    if len(nonzero_coef_vars) != 2:
        # Expect two variables with nonzero cofficient in constraint; otherwise,
        # return empty tuple.
        return ()
    if sorted(coef for coef in rpn.linear_coefs if coef != 0) != [-1, 1]:
        # Expect a constraint of form x == y --> 0 == -1 * x + 1 * y; otherwise,
        # return empty tuple.
        return ()
    # Above checks are satisifed. Return the variables.
    return nonzero_coef_vars


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
    for constraint in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
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
        # Update all elements of set 2 to point to set 1
        for v in set2:
            eq_var_map[v] = set1

    return eq_var_map


class VariableAggregator(IsomorphicTransformation):
    """Aggregate model variables that are linked by equality constraints."""

    alias('contrib.aggregate_vars',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, model):
        """Apply the transformation to the given model."""
        # Generate the equality sets
        # Do the substitution
        # profit
        pass
