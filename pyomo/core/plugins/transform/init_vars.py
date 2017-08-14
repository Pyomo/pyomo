"""Automatically initialize variables."""
from __future__ import division
from pyomo.core.base.var import Var
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias

__author__ = "Qi Chen <qichen at andrew.cmu.edu>"


class InitMidpoint(IsomorphicTransformation):
    """Initializes variables to the midpoint of their bounds."""

    alias('core.init_vars_midpoint', doc=__doc__)

    def __init__(self):
        """Initialize the transformation."""
        super(InitMidpoint, self).__init__()

    def _apply_to(self, instance, overwrite=False):
        """Apply the transformation."""
        for var in instance.component_data_objects(
                ctype=Var, descend_into=True):
            if var.fixed:
                continue
            if var.value is not None and not overwrite:
                continue
            if var.lb is None and var.ub is None:
                # If LB and UB do not exist, set variable value to 0
                var.set_value(0)
            elif var.lb is None:
                # if one bound does not exist, set variable value to the other
                var.set_value(value(var.ub))
            elif var.ub is None:
                # if one bound does not exist, set variable value to the other
                var.set_value(value(var.lb))
            else:
                var.set_value((value(var.lb) + value(var.ub)) / 2)


class InitZero(IsomorphicTransformation):
    """Initializes variables to zeros."""

    alias('core.init_vars_zero', doc=__doc__)

    def __init__(self):
        """Initialize the transformation."""
        super(InitZero, self).__init__()

    def _apply_to(self, instance, overwrite=False):
        """Apply the transformation."""
        for var in instance.component_data_objects(
                ctype=Var, descend_into=True):
            if var.fixed:
                continue
            if var.value is not None and not overwrite:
                continue
            if var.lb is not None and value(var.lb) > 0:
                var.set_value(value(var.lb))
            elif var.ub is not None and value(var.ub) < 0:
                var.set_value(value(var.ub))
            else:
                var.set_value(0)
