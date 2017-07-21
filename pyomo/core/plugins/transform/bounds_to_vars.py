"""Transformation to convert explicit bounds to variable bounds."""
from __future__ import division

from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
import pyomo.core.base.expr as EXPR
from pyomo.repn.canonical_repn import generate_canonical_repn


class ConstraintToVarBoundTransform(IsomorphicTransformation):
    """Change constraints to be a bound on the variable.

    Looks for constraints of form k*v + c1 <= c2. Changes bound on v to match
    (c2 - c1)/k if it results in a tighter bound. Also does the same thing for
    lower bounds.

    """

    alias('core.constraints_to_var_bounds', doc=__doc__)

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
            body = constr.body
            # See if the constraint is a simple x <= b
            try:
                if body.type() is Var:
                    var = body
                    if constr.upper is not None:
                        var.setub(min(var.ub, value(constr.upper))
                                  if var.ub is not None
                                  else value(constr.upper))
                    if constr.lower is not None:
                        var.setlb(max(var.lb, value(constr.lower))
                                  if var.lb is not None
                                  else value(constr.lower))
                constr.deactivate()
                # Sometimes deactivating the constraint will remove the
                # variable from all active constraints, so that it won't be
                # updated during the optimization. Therefore, we need to shift
                # the value of var as necessary in order to keep it within the
                # constraints.
                if var.lb is not None and var.value < var.lb:
                    var.set_value(var.lb)
                if var.ub is not None and var.value > var.ub:
                    var.set_value(var.ub)
                continue
            except AttributeError as err:
                if "object has no attribute 'type'" in err.message:
                    pass
                else:
                    raise
            # Check if the constraint is k * x + c1 <= c2 or c2 <= k * x + c1
            if (isinstance(body, EXPR._SumExpression) and
                    body.polynomial_degree() in (0, 1)):
                repn = generate_canonical_repn(constr.body)
                if repn.variables is not None and len(repn.variables) == 1:
                    var = repn.variables[0]
                    const = repn.constant if repn.constant is not None else 0
                    coef = repn.linear[0]
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
                    # Sometimes deactivating the constraint will remove the
                    # variable from all active constraints, so that it won't be
                    # updated during the optimization. Therefore, we need to
                    # shift the value of var as necessary in order to keep it
                    # within the constraints.
                    if var.lb is not None and var.value < var.lb:
                        var.set_value(var.lb)
                    if var.ub is not None and var.value > var.ub:
                        var.set_value(var.ub)
