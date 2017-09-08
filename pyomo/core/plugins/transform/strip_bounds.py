"""Transformation to strip variable bounds from a model."""
import textwrap

from pyomo.core.base.var import Var
from pyomo.core.base.suffix import Suffix
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias

__author__ = "Qi Chen <https://github.com/qtothec>"


class VariableBoundStripper(IsomorphicTransformation):
    """Strips bounds from variables"""

    alias('core.strip_var_bounds',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(VariableBoundStripper, self).__init__()

    def _apply_to(self, instance, tmp=False):
        """Apply the transformation.

        Args:
            instance (Block): the block on which to strip variable bounds
            tmp (bool, optional): Whether the bound stripping will be
                temporary. If so, store information for reversion.

        Returns:
            None

        """
        if tmp and not hasattr(instance, '_tmp_var_bound_strip_lb'):
            instance._tmp_var_bound_strip_lb = Suffix(direction=Suffix.LOCAL)
        if tmp and not hasattr(instance, '_tmp_var_bound_strip_ub'):
            instance._tmp_var_bound_strip_ub = Suffix(direction=Suffix.LOCAL)
        for var in instance.component_data_objects(ctype=Var):
            if var.has_lb():
                if tmp:
                    instance._tmp_var_bound_strip_lb[var] = var.lb
                var.set_lb(None)
            if var.has_ub():
                if tmp:
                    instance._tmp_var_bound_strip_ub[var] = var.ub
                var.set_ub(None)

    def revert(self, instance):
        """Revert variables fixed by the transformation."""
        for var in instance._tmp_var_bound_strip_lb:
            var.set_lb(instance._tmp_var_bound_strip_lb[var])
        for var in instance._tmp_var_bound_strip_ub:
            var.set_ub(instance._tmp_var_bound_strip_ub[var])
        del instance._tmp_var_bound_strip_lb
        del instance._tmp_var_bound_strip_ub
