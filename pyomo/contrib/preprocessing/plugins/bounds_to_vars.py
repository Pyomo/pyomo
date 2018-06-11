"""Transformation to convert explicit bounds to variable bounds."""
from __future__ import division
import textwrap

from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
from pyomo.repn.canonical_repn import generate_canonical_repn

__author__ = "Qi Chen <https://github.com/qtothec>"


class ConstraintToVarBoundTransform(IsomorphicTransformation):
    """Change constraints to be a bound on the variable.

    Looks for constraints of form k*v + c1 <= c2. Changes bound on v to match
    (c2 - c1)/k if it results in a tighter bound. Also does the same thing for
    lower bounds.

    """

    alias('contrib.constraints_to_var_bounds',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self, *args, **kwargs):
        """Initialize the transformation."""
        super(ConstraintToVarBoundTransform, self).__init__(*args, **kwargs)

    def _create_using(self, model):
        """Create new model, applying transformation."""
        m = model.clone()
        self._apply_to(m)
        return m

    def _apply_to(self, model):
        """Apply the transformation to the given model."""
        m = model

        for constr in m.component_data_objects(ctype=Constraint,
                                               active=True,
                                               descend_into=True):
            # Check if the constraint is k * x + c1 <= c2 or c2 <= k * x + c1
            if constr.body.polynomial_degree() == 1:
                repn = generate_canonical_repn(constr.body)
                if repn.variables is not None and len(repn.variables) == 1:
                    var = repn.variables[0]
                    const = repn.constant if repn.constant is not None else 0
                    coef = float(repn.linear[0])
                    if coef == 0:
                        # This can happen when a one element of a bilinear term
                        # is fixed to zero. Obviously, do not divide by zero.
                        pass
                    else:
                        if constr.upper is not None:
                            newbound = (value(constr.upper) - const) / coef
                            if coef > 0:
                                var.setub(min(var.ub, newbound)
                                          if var.ub is not None
                                          else newbound)
                            elif coef < 0:
                                var.setlb(max(var.lb, newbound)
                                          if var.lb is not None
                                          else newbound)
                        if constr.lower is not None:
                            newbound = (value(constr.lower) - const) / coef
                            if coef > 0:
                                var.setlb(max(var.lb, newbound)
                                          if var.lb is not None
                                          else newbound)
                            elif coef < 0:
                                var.setub(min(var.ub, newbound)
                                          if var.ub is not None
                                          else newbound)
                    constr.deactivate()
                    # Sometimes deactivating the constraint will remove a
                    # variable from all active constraints, so that it won't be
                    # updated during the optimization. Therefore, we need to
                    # shift the value of var as necessary in order to keep it
                    # within its implied bounds, as the constraint we are
                    # deactivating is not an invalid constraint, but rather we
                    # are moving its implied bound directly onto the variable.
                    if (var.has_lb() and var.value is not None
                            and var.value < var.lb):
                        var.set_value(var.lb)
                    if (var.has_ub() and var.value is not None
                            and var.value > var.ub):
                        var.set_value(var.ub)
