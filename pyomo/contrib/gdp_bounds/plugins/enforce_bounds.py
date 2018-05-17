"""Transformation to enforce disjunctive variable bounds."""

from __future__ import division
import textwrap

from pyomo.gdp import Disjunct
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.util.plugin import alias
from pyomo.core import ConstraintList
from six import iteritems


class EnforceDisjunctiveVarBounds(Transformation):
    """Enforces disjunctive variable bounds.

    These are bounds on variables that hold true when an associated disjunct is
    active (indicator_var value is 1). The bounds are enforced using a
    ConstraintList on every Disjunct, enforcing the relevant variable bounds.
    This may lead to duplication of constraints, so the constraints to variable
    bounds preprocessing transformation is recommended for NLP problems
    processed with this transformation.

    """

    alias(
        'contrib.enforce_disj_var_bounds',
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
            if hasattr(disjunct, '_disjunctive_var_constraints'):
                del disjunct._disjunctive_var_constraints
            cons_list = disjunct._disjunctive_var_constraints = ConstraintList()
            for var, bounds in iteritems(disjunct._disj_var_bounds):
                lbb, ubb = bounds
                if lbb is not None:
                    cons_list.add(expr=lbb <= var)
                if ubb is not None:
                    cons_list.add(expr=var <= ubb)
