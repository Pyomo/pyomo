"""Transformation to propagate a zero value to terms of a sum."""
import textwrap

from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.canonical_repn import generate_canonical_repn
from pyomo.util.plugin import alias

__author__ = "Qi Chen <https://github.com/qtothec>"


class ZeroSumPropagator(IsomorphicTransformation):
    """Propagates fixed-to-zero for sums of only positive (or negative) vars.

    If x is fixed to zero and x == x1 + x2 + x3 and x1, x2, x3 are all
    non-negative or all non-positive, then x1, x2, and x3 will be fixed to
    zero.

    """

    alias('contrib.propagate_zero_sum',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(ZeroSumPropagator, self).__init__()

    def _apply_to(self, instance):
        """Apply the transformation.

        Args:
            instance (Block): the block on which to search for x == sum(var)
                constraints. Note that variables may be located anywhere in
                the model.

        Returns:
            None

        """
        for constr in instance.component_data_objects(ctype=Constraint,
                                                      active=True,
                                                      descend_into=True):
            if not constr.body.polynomial_degree() == 1:
                continue  # constraint not linear. Skip.

            repn = generate_canonical_repn(constr.body)
            if (constr.has_ub() and (
                (repn.constant is None and value(constr.upper) == 0) or
                repn.constant == value(constr.upper)
                    )):
                # term1 + term2 + term3 + ... <= 0
                # all var terms need to be non-negative
                if all(
                    # variable has 0 coefficient
                    coef == 0 or
                    # variable is non-negative and has non-negative coefficient
                    (repn.variables[i].has_lb() and
                     value(repn.variables[i].lb) >= 0 and
                     coef >= 0) or
                    # variable is non-positive and has non-positive coefficient
                    (repn.variables[i].has_ub() and
                     value(repn.variables[i].ub) <= 0 and
                     coef <= 0) for i, coef in enumerate(repn.linear)):
                    for i, coef in enumerate(repn.linear):
                        if not coef == 0:
                            repn.variables[i].fix(0)
                    continue
            if (constr.has_lb() and (
                (repn.constant is None and value(constr.lower) == 0) or
                repn.constant == value(constr.lower)
                    )):
                # term1 + term2 + term3 + ... >= 0
                # all var terms need to be non-positive
                if all(
                    # variable has 0 coefficient
                    coef == 0 or
                    # variable is non-negative and has non-positive coefficient
                    (repn.variables[i].has_lb() and
                     value(repn.variables[i].lb) >= 0 and
                     coef <= 0) or
                    # variable is non-positive and has non-negative coefficient
                    (repn.variables[i].has_ub() and
                     value(repn.variables[i].ub) <= 0 and
                     coef >= 0) for i, coef in enumerate(repn.linear)):
                    for i, coef in enumerate(repn.linear):
                        if not coef == 0:
                            repn.variables[i].fix(0)
