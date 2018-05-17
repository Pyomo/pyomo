"""Transformation to aggregate equal variables."""

from __future__ import division
import textwrap

from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
from pyomo.repn import generate_standard_repn


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
