"""Transformation to reformulate nonlinear models with linearity induced from
discrete variables.

Ref: Grossmann, IE; Voudouris, VT; Ghattas, O. Mixed integer linear
reformulations for some nonlinear discrete design optimization problems.

"""

from __future__ import division

import textwrap

from pyomo.core.base import Block, Constraint, VarList, Objective
from pyomo.core.expr.current import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
from pyomo.common.plugin import alias
import logging

logger = logging.getLogger('pyomo.contrib.preprocessing')


class InducedLinearity(IsomorphicTransformation):
    """Reformulate nonlinear constraints with induced linearity.

    Finds continuous variables v where v = d1 + d2 + d3, where d's are discrete
    variables. These continuous variables may participate nonlinearly in other
    expressions, which may then be induced to be linear.

    """

    alias('contrib.induced_linearity',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, model):
        """Apply the transformation to the given model."""
        pass
        # detect variables that are effectively discrete because they are the
        # sum of discrete variables
        pass
        # Find if these variables participate in a bilinear term
        pass
        # Reformulate the bilinear terms
