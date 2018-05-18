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
