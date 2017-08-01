"""Transformation to remove zero terms from constraints."""
from __future__ import division

from pyomo.core.base.constraint import Constraint
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
from pyomo.repn.canonical_repn import generate_canonical_repn


class RemoveZeroTerms(IsomorphicTransformation):
    """Looks for 0 * var in a constraint and removes it.

    Currently limited to processing linear constraints of the form x1 = 0 *
    x3, occurring as a result of x2.fix(0).

    """

    alias('core.remove_zero_terms', doc=__doc__)

    def __init__(self, *args, **kwargs):
        """Initialize the transformation."""
        super(RemoveZeroTerms, self).__init__(*args, **kwargs)

    def _apply_to(self, model):
        """Apply the transformation."""
        m = model

        for constr in m.component_data_objects(ctype=Constraint,
                                               active=True,
                                               descend_into=True):
            if not constr.body.polynomial_degree() == 1:
                continue  # we currently only process linear constraints
            repn = generate_canonical_repn(constr.body)
            nonzero_vars_indx = [i for i in range(len(repn.variables))
                                 if not repn.linear[i] == 0]
            const = repn.constant if repn.constant is not None else 0

            if constr.equality:
                constr.set_value(sum(repn.linear[i] * repn.variables[i]
                                     for i in nonzero_vars_indx) + const ==
                                 constr.upper)
            elif constr.lower is not None:
                constr.set_value(sum(repn.linear[i] * repn.variables[i]
                                     for i in nonzero_vars_indx) + const >=
                                 constr.lower)
            elif constr.upper is not None:
                constr.set_value(sum(repn.linear[i] * repn.variables[i]
                                     for i in nonzero_vars_indx) + const <=
                                 constr.upper)
