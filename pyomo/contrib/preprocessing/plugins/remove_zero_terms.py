# -*- coding: UTF-8 -*-
"""Transformation to remove zero terms from constraints."""
from __future__ import division

import textwrap

from pyomo.core import quicksum
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr import current as EXPR
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
from pyomo.common.plugin import alias
from pyutilib.math.util import isclose


class RemoveZeroTerms(IsomorphicTransformation):
    """Looks for 0 * var in a constraint and removes it.

    Currently limited to processing linear constraints of the form x1 = 0 *
    x3, occurring as a result of x2.fix(0).

    TODO: support nonlinear expressions

    """

    alias(
        'contrib.remove_zero_terms',
        doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, model):
        """Apply the transformation."""
        m = model

        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            if not constr.body.polynomial_degree() == 1:
                continue  # we currently only process linear constraints
            repn = generate_standard_repn(constr.body)

            # get the index of all nonzero coefficient variables
            nonzero_vars_indx = [
                i for i, _ in enumerate(repn.linear_vars)
                if not repn.linear_coefs[i] == 0
            ]
            const = repn.constant

            # reconstitute the constraint, including only variable terms with
            # nonzero coefficients
            constr_body = quicksum(repn.linear_coefs[i] * repn.linear_vars[i]
                                   for i in nonzero_vars_indx) + const
            if constr.equality:
                constr.set_value(constr_body == constr.upper)
            elif constr.has_lb() and not constr.has_ub():
                constr.set_value(constr_body >= constr.lower)
            elif constr.has_ub() and not constr.has_lb():
                constr.set_value(constr_body <= constr.upper)
            else:
                # constraint is a bounded inequality of form a <= x <= b.
                constr.set_value(EXPR.inequality(
                    constr.lower, constr_body, constr.upper))
