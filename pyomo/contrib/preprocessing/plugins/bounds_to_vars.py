#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Transformation to convert explicit bounds to variable bounds."""

from __future__ import division

from math import fabs
import math

from pyomo.core.base.transformation import TransformationFactory
from pyomo.common.config import (
    ConfigBlock,
    ConfigValue,
    NonNegativeFloat,
    document_kwargs_from_configdict,
)
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn


@TransformationFactory.register(
    'contrib.constraints_to_var_bounds',
    doc="Change constraints to be a bound on the variable.",
)
@document_kwargs_from_configdict('CONFIG')
class ConstraintToVarBoundTransform(IsomorphicTransformation):
    """Change constraints to be a bound on the variable.

    Looks for constraints of form: :math:`k*v + c_1 \\leq c_2`. Changes
    variable lower bound on :math:`v` to match :math:`(c_2 - c_1)/k` if it
    results in a tighter bound. Also does the same thing for lower bounds.

    Keyword arguments below are specified for the ``apply_to`` and
    ``create_using`` functions.

    """

    CONFIG = ConfigBlock("ConstraintToVarBounds")
    CONFIG.declare(
        "tolerance",
        ConfigValue(
            default=1e-13,
            domain=NonNegativeFloat,
            description="tolerance on bound equality (:math:`LB = UB`)",
        ),
    )
    CONFIG.declare(
        "detect_fixed",
        ConfigValue(
            default=True,
            domain=bool,
            description="If True, fix variable when "
            ":math:`| LB - UB | \\leq tolerance`.",
        ),
    )

    def _apply_to(self, model, **kwds):
        config = self.CONFIG(kwds)

        for constr in model.component_data_objects(
            ctype=Constraint, active=True, descend_into=True
        ):
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
                    new_ub = (constr.ub - const) / coef
                    var_ub = float('inf') if var.ub is None else var.ub
                    var.setub(min(var_ub, new_ub))
                if constr.has_lb():
                    new_lb = (constr.lb - const) / coef
                    var_lb = float('-inf') if var.lb is None else var.lb
                    var.setlb(max(var_lb, new_lb))
            elif coef < 0:
                if constr.has_ub():
                    new_lb = (constr.ub - const) / coef
                    var_lb = float('-inf') if var.lb is None else var.lb
                    var.setlb(max(var_lb, new_lb))
                if constr.has_lb():
                    new_ub = (constr.lb - const) / coef
                    var_ub = float('inf') if var.ub is None else var.ub
                    var.setub(min(var_ub, new_ub))

            if var.is_integer():
                # Make sure that the lb and ub are integral. Use safe
                # construction if near to integer.
                if var.has_lb():
                    var.setlb(
                        int(
                            min(math.ceil(var.lb - config.tolerance), math.ceil(var.lb))
                        )
                    )
                if var.has_ub():
                    var.setub(
                        int(
                            max(
                                math.floor(var.ub + config.tolerance),
                                math.floor(var.ub),
                            )
                        )
                    )

            if var is not None and var.value is not None:
                _adjust_var_value_if_not_feasible(var)

            if config.detect_fixed and var.has_lb() and var.has_ub():
                lb, ub = var.bounds
                if lb == ub:
                    var.fix(lb)
                elif fabs(lb - ub) <= config.tolerance:
                    # If the bounds are note exactly equal, set the
                    # value to the midpoint
                    var.fix((lb + ub) / 2)

            constr.deactivate()


def _adjust_var_value_if_not_feasible(var):
    # Sometimes deactivating the constraint will remove a
    # variable from all active constraints, so that it won't be
    # updated during the optimization. Therefore, we need to
    # shift the value of var as necessary in order to keep it
    # within its implied bounds, as the constraint we are
    # deactivating is not an invalid constraint, but rather we
    # are moving its implied bound directly onto the variable.
    var_value = var.value
    if var.has_lb():
        var_value = max(var_value, var.lb)
    if var.has_ub():
        var_value = min(var_value, var.ub)
    if var.is_integer():
        var.set_value(int(var_value))
    else:
        var.set_value(var_value)
