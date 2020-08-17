#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Transformation to detect variables fixed by bounds and fix them."""
from math import fabs

from six import iteritems

from pyomo.core.base.plugin import TransformationFactory
from pyomo.common.collections import ComponentMap
from pyomo.common.config import (ConfigBlock, ConfigValue, NonNegativeFloat,
                                 add_docstring_list)
from pyomo.core.base.var import Var
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation


@TransformationFactory.register(
        'contrib.detect_fixed_vars',
        doc="Detect variables that are de-facto fixed but not considered fixed.")
class FixedVarDetector(IsomorphicTransformation):
    """Detects variables that are de-facto fixed but not considered fixed.

    For each variable :math:`v` found on the model, check to see if its lower
    bound :math:`v^{LB}` is within some tolerance of its upper bound
    :math:`v^{UB}`. If so, fix the variable to the value of :math:`v^{LB}`.

    Keyword arguments below are specified for the ``apply_to`` and
    ``create_using`` functions.

    """

    CONFIG = ConfigBlock("FixedVarDetector")
    CONFIG.declare("tmp", ConfigValue(
        default=False, domain=bool,
        description="True to store the set of transformed variables and "
        "their old values so that they can be restored."
    ))
    CONFIG.declare("tolerance", ConfigValue(
        default=1E-13, domain=NonNegativeFloat,
        description="tolerance on bound equality (LB == UB)"
    ))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    def _apply_to(self, instance, **kwargs):
        config = self.CONFIG(kwargs)

        if config.tmp:
            instance._xfrm_detect_fixed_vars_old_values = ComponentMap()

        for var in instance.component_data_objects(
                ctype=Var, descend_into=True):
            if var.fixed or var.lb is None or var.ub is None:
                # if the variable is already fixed, or if it is missing a
                # bound, we skip it.
                continue
            if fabs(value(var.lb) - value(var.ub)) <= config.tolerance:
                if config.tmp:
                    instance._xfrm_detect_fixed_vars_old_values[var] = \
                        var.value
                var.fix(var.lb)

    def revert(self, instance):
        """Revert variables fixed by the transformation."""
        for var, var_value in iteritems(
                instance._xfrm_detect_fixed_vars_old_values):
            var.unfix()
            var.set_value(var_value)

        del instance._xfrm_detect_fixed_vars_old_values
