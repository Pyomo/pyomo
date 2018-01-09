"""Transformation to detect variables fixed by bounds and fix them."""
import textwrap
from math import fabs

from six import iteritems

from pyomo.core.base.var import Var
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
from pyomo.core.kernel.component_map import ComponentMap


class FixedVarDetector(IsomorphicTransformation):
    """Detects variables that are de-facto fixed but not considered fixed.

    Descends through the model. For each variable found, check to see if var.lb
    is within some tolerance of var.ub. If so, fix the variable to the value of
    var.lb.

    """

    alias(
        'contrib.detect_fixed_vars',
        doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(FixedVarDetector, self).__init__()

    def _apply_to(self, instance, **kwargs):
        """Apply the transformation.

        Args:
            instance: Pyomo model object to transform.

        Kwargs:
            tmp: True to store the set of transformed variables and their old
                values so that they can be restored.
            tol: tolerance on bound equality (LB == UB)
        """
        tmp = kwargs.pop('tmp', False)
        tol = kwargs.pop('tolerance', 1E-13)

        if tmp:
            instance._xfrm_detect_fixed_vars_old_values = ComponentMap()

        for var in instance.component_data_objects(
                ctype=Var, descend_into=True):
            if var.fixed or var.lb is None or var.ub is None:
                # if the variable is already fixed, or if it is missing a
                # bound, we skip it.
                continue
            if fabs(value(var.lb - var.ub)) <= tol:
                if tmp:
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
