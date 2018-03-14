# -*- coding: utf-8 -*-
"""Transformation to deactivate trivial constraints."""
import textwrap
from math import fabs

from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias


class TrivialConstraintDeactivator(IsomorphicTransformation):
    """Deactivates trivial constraints.

    These are constraints of form constant = constant or constant <= constant.
    These constraints typically arise when variables are fixed.

    """

    alias(
        'contrib.deactivate_trivial_constraints',
        doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(TrivialConstraintDeactivator, self).__init__()

    def _apply_to(self, instance, **kwargs):
        """Apply the transformation.

        Args:
            instance: Pyomo model object to transform.

        Kwargs:
            tmp: True to store a set of transformed constraints for future
                reversion of the transformation
            ignore_infeasible: True to skip over trivial constraints that are
                infeasible instead of raising a ValueError.
            return_trivial: a list to which the deactivated trivial
                constraints are appended (side effect)
            tolerance: tolerance on constraint violations
        """
        tmp = kwargs.pop('tmp', False)
        ignore_infeasible = kwargs.pop('ignore_infeasible', False)
        tol = kwargs.pop('tolerance', 1E-13)
        trivial = kwargs.pop('return_trivial', [])
        if tmp and not hasattr(instance, '_tmp_trivial_deactivated_constrs'):
            instance._tmp_trivial_deactivated_constrs = ComponentSet()

        # Trivial constraints are those that do not contain any variables, ie.
        # the polynomial degree is 0
        trivial_constraints = (
            c
            for c in instance.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
            if c.body.polynomial_degree() == 0)

        for constr in trivial_constraints:
            # We need to check each constraint to sure that it is not violated.

            # Check if the lower bound is violated outside a given tolerance
            if (constr.has_lb()
                    and value(constr.body) + tol <= value(constr.lower)):
                # Trivial constraint is infeasible.
                if ignore_infeasible:
                    # do nothing, move on to next constraint
                    continue
                else:
                    raise ValueError(
                        'Trivial constraint {} violates LB {} ≤ BODY {}.'
                        .format(constr.name, value(constr.lower),
                                value(constr.body)))

            # Check if the upper bound is violated outside a given tolerance
            if (constr.has_ub()
                    and value(constr.body) >= value(constr.upper) + tol):
                # Trivial constraint is infeasible.
                if ignore_infeasible:
                    # do nothing, move on to next constraint
                    continue
                else:
                    raise ValueError(
                        'Trivial constraint {} violates BODY {} ≤ UB {}.'
                        .format(constr.name, value(constr.body),
                                value(constr.upper)))

            # Constraint is not infeasible. Deactivate it.
            if tmp:
                instance._tmp_trivial_deactivated_constrs.add(constr)
            trivial.append(constr)
            constr.deactivate()

    def revert(self, instance):
        """Revert constraints deactivated by the transformation.

        Args:
            instance: the model instance on which trivial constraints were
                earlier deactivated.
        """
        for constr in instance._tmp_trivial_deactivated_constrs:
            constr.activate()
        del instance._tmp_trivial_deactivated_constrs
