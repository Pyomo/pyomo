"""Transformation to enable disjunctive variable bounds."""

from __future__ import division
import textwrap

from pyomo.gdp import Disjunct
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.util.plugin import alias
from pyomo.core import ConstraintList


class EnableDisjunctiveVarBounds(Transformation):
    """Enables disjunctive variable bounds.

    These are bounds on variables that hold true when an associated disjunct
    is active (indicator_var value is 1).

    """

    alias(
        'contrib.enable_disjunctive_bounds',
        doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, scope):
        """Apply the transformation.

        Args:
            scope: Pyomo model object to transform.

        """
        disjuncts_to_process = list(scope.component_data_objects(
            ctype=Disjunct, active=True, descend_into=(Block, Disjunct),
            descent_order=TraversalStrategy.BreadthFirstSearch))
        if scope.type() == Disjunct:
            disjuncts_to_process.insert(0, scope)

        for disjunct in disjuncts_to_process:
            if not hasattr(disjunct, '_disjunctive_bounds'):
                disjunct._disjunctive_bounds = ComponentMap()

            if not hasattr(disjunct, 'disjunctive_var_constraints'):
                disjunct.disjunctive_var_constraints = ConstraintList()
