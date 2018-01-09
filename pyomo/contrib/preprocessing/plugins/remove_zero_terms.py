# -*- coding: UTF-8 -*-
"""Transformation to remove zero terms from constraints."""
from __future__ import division

import textwrap

from pyomo.core.base.constraint import Constraint
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.util.plugin import alias


class RemoveZeroTerms(IsomorphicTransformation):
    """Looks for 0 * var in a constraint and removes it.

    Currently limited to processing linear constraints of the form x1 = 0 *
    x3, occurring as a result of x2.fix(0).

    TODO: support nonlinear expressions

    """

    alias(
        'contrib.remove_zero_terms',
        doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self, *args, **kwargs):
        """Initialize the transformation."""
        super(RemoveZeroTerms, self).__init__(*args, **kwargs)

    def _apply_to(self, model):
        """Apply the transformation."""
        m = model

        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            if not constr.body.polynomial_degree() == 1:
                continue  # we currently only process linear constraints
            repn = generate_canonical_repn(constr.body)

            # get the index of all nonzero coefficient variables
            nonzero_vars_indx = [
                i for i, _ in enumerate(repn.variables)
                if not repn.linear[i] == 0
            ]
            const = repn.constant if repn.constant is not None else 0

            # reconstitute the constraint, including only variable terms with
            # nonzero coefficients
            constr_body = sum(repn.linear[i] * repn.variables[i]
                              for i in nonzero_vars_indx) + const
            if constr.equality:
                constr.set_value(constr_body == constr.upper)
            elif constr.has_lb() and not constr.has_ub():
                constr.set_value(constr_body >= constr.lower)
            elif constr.has_ub() and not constr.has_lb():
                constr.set_value(constr_body <= constr.upper)
            else:
                # constraint is a bounded inequality of form a <= x <= b.
                # I don't think this is a great idea, but ¯\_(ツ)_/¯
                constr.set_value(constr.lower <= constr_body <= constr.upper)
