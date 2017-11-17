# -*- coding: UTF-8 -*-
"""Transformation to deactivate trivial constraints."""
import textwrap
from math import fabs

from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias

__author__ = "Qi Chen <https://github.com/qtothec>"


class TrivialConstraintDeactivator(IsomorphicTransformation):
    """Deactivates trivial constraints of form constant = constant."""

    alias('core.deactivate_trivial_constraints',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(TrivialConstraintDeactivator, self).__init__()
        self.tolerance = 1E-14

    def _apply_to(self, instance, tmp=False):
        """Apply the transformation."""
        if tmp and not hasattr(instance, '_tmp_trivial_deactivated_constrs'):
            instance._tmp_trivial_deactivated_constrs = ComponentSet()

        for constr in instance.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            if constr.body.polynomial_degree() == 0:
                # Check to make sure constraint not violated.
                if (constr.has_lb() and
                        value(constr.body) < value(constr.lower)):
                    # Sometimes if values are close to zero, but not quite
                    # zero, we run into issues. From a practical perspective,
                    # let's apply a tolerance.
                    if (fabs(value(constr.body)) <= self.tolerance and
                            fabs(value(constr.lower)) <= self.tolerance):
                        pass
                    else:
                        raise ValueError(
                            'Trivial constraint {} violates {} ≤ {}.'
                            .format(constr.name,
                                    value(constr.lower),
                                    value(constr.body)))
                if (constr.has_ub() and
                        value(constr.body) > value(constr.upper)):
                    if (fabs(value(constr.body)) <= self.tolerance and
                            fabs(value(constr.upper)) <= self.tolerance):
                        pass
                    else:
                        raise ValueError(
                            'Trivial constraint {} violates {} ≤ {}.'
                            .format(constr.name,
                                    value(constr.body),
                                    value(constr.upper)))
                # Constraint is fine. Deactivate it.
                if tmp:
                    instance._tmp_trivial_deactivated_constrs.add(constr)
                constr.deactivate()

    def revert(self, instance):
        """Revert constraints deactivated by the transformation."""
        for constr in instance._tmp_trivial_deactivated_constrs:
            constr.activate()
        del instance._tmp_trivial_deactivated_constrs
