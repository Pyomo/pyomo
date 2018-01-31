"""Transformation to strip variable bounds from a model."""
import textwrap

from pyomo.core.base.var import Var
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation
from pyomo.util.plugin import alias
from pyomo.core.kernel.component_map import ComponentMap
from six import iteritems
from pyomo.core.kernel.set_types import Reals


class VariableBoundStripper(NonIsomorphicTransformation):
    """Strips bounds from variables."""

    alias('core.strip_var_bounds',
          doc=textwrap.fill(textwrap.dedent(__doc__.strip())))

    def __init__(self):
        """Initialize the transformation."""
        super(VariableBoundStripper, self).__init__()

    def _apply_to(self, instance, strip_domains=True, reversible=False):
        """Apply the transformation.

        Args:
            instance (Block): the block on which to strip variable bounds
            strip_domains (bool, optional): strip the domain for discrete
                variables as well
            reversible (bool, optional): Whether the bound stripping will be
                temporary. If so, store information for reversion.

        Returns:
            None

        """
        if reversible:
            # Component maps to store data for reversion. Pyomo should warn if
            # a map already exists.
            instance._tmp_var_bound_strip_lb = ComponentMap()
            instance._tmp_var_bound_strip_ub = ComponentMap()
            instance._tmp_var_bound_strip_domain = ComponentMap()
        for var in instance.component_data_objects(ctype=Var):
            if strip_domains and not var.domain == Reals:
                if reversible:
                    instance._tmp_var_bound_strip_domain[var] = var.domain
                var.domain = Reals
            if var.has_lb():
                if reversible:
                    instance._tmp_var_bound_strip_lb[var] = var.lb
                var.setlb(None)
            if var.has_ub():
                if reversible:
                    instance._tmp_var_bound_strip_ub[var] = var.ub
                var.setub(None)

    def revert(self, instance):
        """Revert variable bounds and domains changed by the transformation."""
        for var, lb in iteritems(instance._tmp_var_bound_strip_lb):
            var.setlb(lb)
        for var, ub in iteritems(instance._tmp_var_bound_strip_ub):
            var.setub(ub)
        for var, dom in iteritems(instance._tmp_var_bound_strip_domain):
            var.domain = dom
        del instance._tmp_var_bound_strip_lb
        del instance._tmp_var_bound_strip_ub
        del instance._tmp_var_bound_strip_domain
