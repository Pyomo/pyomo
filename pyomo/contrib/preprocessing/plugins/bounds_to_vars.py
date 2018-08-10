"""Transformation to convert explicit bounds to variable bounds."""

from __future__ import division

import textwrap

from pyomo.common.plugin import alias
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
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
        for constr in model.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            # Check if the constraint is k * x + c1 <= c2 or c2 <= k * x + c1
            repn = generate_standard_repn(constr.body)
            if not repn.is_linear() or len(repn.linear_vars) != 1:
                # Skip nonlinear constraints, trivial constraints, and those
                # that involve more than one variable.
                continue
            else:
                var = repn.linear_vars[0]
                const = repn.constant
                coef = float(repn.linear_coefs[0])

            if coef == 0:
                # Skip trivial constraints
                continue
            elif coef > 0:
                if constr.has_ub():
                    new_ub = (value(constr.upper) - const) / coef
                    var_ub = float('inf') if var.ub is None else var.ub
                    var.setub(min(var_ub, new_ub))
                if constr.has_lb():
                    new_lb = (value(constr.lower) - const) / coef
                    var_lb = float('-inf') if var.lb is None else var.lb
                    var.setlb(max(var_lb, new_lb))
            elif coef < 0:
                if constr.has_ub():
                    new_lb = (value(constr.upper) - const) / coef
                    var_lb = float('-inf') if var.lb is None else var.lb
                    var.setlb(max(var_lb, new_lb))
                if constr.has_lb():
                    new_ub = (value(constr.lower) - const) / coef
                    var_ub = float('inf') if var.ub is None else var.ub
                    var.setub(min(var_ub, new_ub))

            if var is not None and var.value is not None:
                # Sometimes deactivating the constraint will remove a
                # variable from all active constraints, so that it won't be
                # updated during the optimization. Therefore, we need to
                # shift the value of var as necessary in order to keep it
                # within its implied bounds, as the constraint we are
                # deactivating is not an invalid constraint, but rather we
                # are moving its implied bound directly onto the variable.
                if var.has_lb():
                    var_value = max(var.value, var.lb)
                if var.has_ub():
                    var_value = min(var.value, var.ub)
                var.set_value(var_value)

            constr.deactivate()
