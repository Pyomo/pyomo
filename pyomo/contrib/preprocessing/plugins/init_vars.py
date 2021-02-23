#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Automatically initialize variables."""
from __future__ import division

from pyomo.core.base.var import Var
from pyomo.core.base.plugin import TransformationFactory
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation


@TransformationFactory.register(
        'contrib.init_vars_midpoint',
        doc="Initialize non-fixed variables to the midpoint of their bounds.")
class InitMidpoint(IsomorphicTransformation):
    """Initialize non-fixed variables to the midpoint of their bounds.

    - If the variable does not have bounds, set the value to zero.
    - If the variable is missing one bound, set the value to that of the
      existing bound.
    """

    def _apply_to(self, instance, overwrite=False):
        """Apply the transformation.

        Kwargs:
            overwrite: if False, transformation will not overwrite existing
                variable values.
        """
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
                var.set_value((value(var.lb) + value(var.ub)) / 2.)


@TransformationFactory.register(
        'contrib.init_vars_zero',
        doc="Initialize non-fixed variables to zero.")
class InitZero(IsomorphicTransformation):
    """Initialize non-fixed variables to zero.

    - If setting the variable value to zero will violate a bound, set the
      variable value to the relevant bound value.

    """

    def _apply_to(self, instance, overwrite=False):
        """Apply the transformation.

        Kwargs:
            overwrite: if False, transformation will not overwrite existing
                variable values.
        """
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
