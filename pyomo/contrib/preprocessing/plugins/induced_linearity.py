"""Transformation to reformulate nonlinear models with linearity induced from
discrete variables.

Ref: Grossmann, IE; Voudouris, VT; Ghattas, O. Mixed integer linear
reformulations for some nonlinear discrete design optimization problems.

"""

from __future__ import division

import textwrap
from math import fabs

from pyomo.core.base import Block, Constraint, VarList, Objective
from pyomo.core.expr.current import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.kernel import ComponentMap, ComponentSet
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
from pyomo.common.plugin import alias
from pyomo.gdp import Disjunct
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
        constraint_bound_tolerance = 1E-6
        effectively_discrete_vars = ComponentSet(
            _effectively_discrete_vars(model, constraint_bound_tolerance))
        # TODO will need to go through this for each disjunct

        # Collect find bilinear expressions that can be reformulated using
        # knowledge of effectively discrete variables
        pass
        # Categorize the constraint as Case 1 or Case 2
        pass
        # Reformulate the bilinear terms
        pass


def _bilinear_expressions(model):
    # TODO for now, we look for only expressions where the bilinearities are
    # exposed on the root level SumExpression, and thus accessible via
    # generate_standard_repn. This will not detect exp(x*y). We require a
    # factorization transformation to be applied beforehand in order to pick
    # these constraints up.
    pass
    for constr in model.component_data_objects(
            Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() in (1, 0):
            continue  # Skip trivial and linear constraints
        repn = generate_standard_repn(constr.body)
        pass
        # TODO what information do I need to store?


def _effectively_discrete_vars(block, constraint_bound_tolerance):
    """Yield variables that are effectively discrete because they are the
    sum of discrete variables.
    """
    for constr in block.component_data_objects(
            Constraint, active=True, descend_into=True):
        if constr.lower is None or constr.upper is None:
            continue  # skip inequality constraints
        if fabs(value(constr.lower) - value(constr.upper)
                ) > constraint_bound_tolerance:
            continue  # not equality constriant. Skip.
        if constr.body.polynomial_degree() not in (1, 0):
            continue  # skip nonlinear expressions
        repn = generate_standard_repn(constr.body)
        if len(repn.linear_vars) < 2:
            # TODO we should make sure that trivial equality relations are
            # preprocessed before this, or we will end up reformulating
            # expressions that we do not need to here.
            continue
        non_discrete_vars = list(v for v in repn.linear_vars
                                 if v.is_continuous())
        if len(non_discrete_vars) == 1:
            # We know that this is an effectively discrete continuous
            # variable. Add it to our identified variable list.
            yield non_discrete_vars[0]
