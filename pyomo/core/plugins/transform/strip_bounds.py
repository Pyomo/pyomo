"""Transformation to strip variable bounds from a model."""
import textwrap

from pyomo.core.base.var import Var
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.util.plugin import alias
from pyomo.core.kernel.component_map import ComponentMap
from six import iteritems
from pyomo.core.kernel.set_types import Reals

__author__ = "Qi Chen <https://github.com/qtothec>"


class VariableBoundStripper(IsomorphicTransformation):
    """Strips bounds from variables."""

    alias('core.strip_var_bounds',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(VariableBoundStripper, self).__init__()

    def _apply_to(self, instance, strip_domains=True, tmp=False):
        """Apply the transformation.

        Args:
            instance (Block): the block on which to strip variable bounds
            strip_domains (bool, optional): strip the domain for discrete
                variables as well
            tmp (bool, optional): Whether the bound stripping will be
                temporary. If so, store information for reversion.

        Returns:
            None

        """
        if tmp and not hasattr(instance, '_tmp_var_bound_strip_lb'):
            instance._tmp_var_bound_strip_lb = ComponentMap()
        if tmp and not hasattr(instance, '_tmp_var_bound_strip_ub'):
            instance._tmp_var_bound_strip_ub = ComponentMap()
        if tmp and not hasattr(instance, '_tmp_var_bound_strip_domain'):
            instance._tmp_var_bound_strip_domain = ComponentMap()
        for var in instance.component_data_objects(ctype=Var):
            if strip_domains and not var.domain == Reals:
                if tmp:
                    instance._tmp_var_bound_strip_domain[var] = var.domain
                var.domain = Reals
            if var.has_lb():
                if tmp:
                    instance._tmp_var_bound_strip_lb[var] = var.lb
                var.setlb(None)
            if var.has_ub():
                if tmp:
                    instance._tmp_var_bound_strip_ub[var] = var.ub
                var.setub(None)

    def revert(self, instance):
        """Revert variables fixed by the transformation."""
        for var, lb in iteritems(instance._tmp_var_bound_strip_lb):
            var.setlb(lb)
        for var, ub in iteritems(instance._tmp_var_bound_strip_ub):
            var.setub(ub)
        for var, dom in iteritems(instance._tmp_var_bound_strip_domain):
            var.domain = dom
        del instance._tmp_var_bound_strip_lb
        del instance._tmp_var_bound_strip_ub
        del instance._tmp_var_bound_strip_domain
