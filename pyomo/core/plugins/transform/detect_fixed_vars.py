"""Transformation to detect variables fixed by bounds and fix them."""
from pyomo.core.base.block import generate_cuid_names
from pyomo.core.base.var import Var
from pyomo.core.kernel.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
from six import iteritems

__author__ = "Qi Chen <qichen at andrew.cmu.edu>"


class FixedVarDetector(IsomorphicTransformation):
    """Detects variables that are de-facto fixed but not considered fixed.

    Checks to see if var.lb == var.ub. If so, fixes the variable to the value
    of var.lb.

    """

    alias('core.detect_fixed_vars', doc=__doc__)

    def __init__(self):
        """Initialize the transformation."""
        super(FixedVarDetector, self).__init__()
        self._old_var_values = {}
        self._transformed_instance = None

    def _apply_to(self, instance, tmp=False):
        """Apply the transformation."""
        if tmp:
            self._transformed_instance = instance

        #: dict: Mapping of variable to its UID
        var_to_id = self.var_to_id = generate_cuid_names(
            instance.model(), ctype=Var, descend_into=True)
        #: dict: Mapping of UIDs to variables
        self.id_to_var = dict(
            (cuid, obj) for obj, cuid in iteritems(var_to_id))

        for var in instance.component_data_objects(
                ctype=Var, descend_into=True):
            if var.fixed or var.lb is None or var.ub is None:
                continue
            if value(var.lb) == value(var.ub):
                self._old_var_values[var_to_id[var]] = var.value
                var.fix(var.lb)

    def revert(self):
        """Revert variables fixed by the transformation."""
        for varUID in self._old_var_values:
            var = self.id_to_var[varUID]
            var.unfix()
            var.set_value(self._old_var_values[varUID])
        self._old_var_values = {}
