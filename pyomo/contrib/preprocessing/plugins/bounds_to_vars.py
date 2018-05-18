"""Transformation to convert explicit bounds to variable bounds."""

from __future__ import division

import textwrap

from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.common.plugin import alias
from pyomo.repn import generate_standard_repn


class ConstraintToVarBoundTransform(IsomorphicTransformation):
    """Change constraints to be a bound on the variable.

    Looks for constraints of form k*v + c1 <= c2. Changes bound on v to match
    (c2 - c1)/k if it results in a tighter bound. Also does the same thing for
    lower bounds.
    """

    alias('contrib.constraints_to_var_bounds',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def _apply_to(self, model):
        """Apply the transformation to the given model."""
        m = model

        for constr in m.component_data_objects(ctype=Constraint,
                                               active=True,
                                               descend_into=True):
            # Check if the constraint is k * x + c1 <= c2 or c2 <= k * x + c1
            repn = generate_standard_repn(constr.body)
            if repn.is_linear():
                if len(repn.linear_vars) == 0:
                    var = None
                    const = repn.constant
                    coef = 0
                elif len(repn.linear_vars) == 1:
                    var = repn.linear_vars[0]
                    const = repn.constant
                    coef = float(repn.linear_coefs[0])
                else:
                    continue
                if coef == 0:
                    # If a value is zero, we ignore it so we don't
                    # raise a divide-by-dero exception.  We can
                    # safely deactivate this constraint as long as
                    # the constant is within the constraint bounds.
                    if constr.upper is not None \
                       and (value(constr.upper) - const) < 0:
                        # This constraint is trivially infeasible.
                        # We need to leave it in the model so that
                        # the writer complains.
                        continue
                    if constr.lower is not None \
                       and (value(constr.lower) - const) > 0:
                        # This constraint is trivially infeasible.
                        # We need to leave it in the model so that
                        # the writer complains.
                        continue
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

                constr.deactivate()
