# -*- coding: UTF-8 -*-
"""Transformation to deactivate trivial constraints."""
from pyomo.core.base.constraint import Constraint
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias

__author__ = "Qi Chen <qichen at andrew.cmu.edu>"


class TrivialConstraintDeactivator(IsomorphicTransformation):
    """Deactivates trivial constraints of form constant = constant."""

    alias('core.deactivate_trivial_constraints', doc=__doc__)

    def __init__(self):
        """Initialize the transformation."""
        super(TrivialConstraintDeactivator, self).__init__()
        self._deactivated_constrs = set()
        self._transformed_instance = None

    def _apply_to(self, instance, tmp=False):
        """Apply the transformation."""
        if tmp:
            self._transformed_instance = instance

        for constr in instance.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            if constr.body.polynomial_degree() == 0:
                # Check to make sure constraint not violated.
                if (constr.has_lb() and
                        value(constr.body) < value(constr.lower)):
                    raise ValueError('Trivial constraint {} violates {} ≤ {}'
                                     .format(constr.name,
                                             value(constr.lower),
                                             value(constr.body)))
                if (constr.has_ub() and
                        value(constr.body) > value(constr.upper)):
                    raise ValueError('Trivial constraint {} violates {} ≤ {}'
                                     .format(constr.name,
                                             value(constr.body),
                                             value(constr.upper)))
                # Constraint is fine. Deactivate it.
                self._deactivated_constrs.add(constr)
                constr.deactivate()

    def revert(self):
        """Revert constraints deactivated by the transformation."""
        for constr in self._deactivated_constrs:
            constr.activate()
        self._deactivated_constrs.clear()
