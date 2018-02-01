"""Transformation to strip variable bounds from a model."""
import textwrap

from pyomo.core.base.var import Var
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation
from pyomo.util.plugin import alias
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.set_types import Reals


class VariableBoundStripper(NonIsomorphicTransformation):
    """Strips bounds from variables."""

    alias('contrib.strip_var_bounds',
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
            if any(hasattr(instance, map_name) for map_name in [
                    '_tmp_var_bound_strip_lb',
                    '_tmp_var_bound_strip_ub',
                    '_tmp_var_bound_strip_domain']):
                raise RuntimeError(
                    'Variable stripping reversion component maps already '
                    'exist. Did you already apply a temporary transformation '
                    'without a subsequent reversion?')
            # Component maps to store data for reversion.
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
        for var in instance.component_data_objects(
                ctype=Var, descend_into=True):
            if var in instance._tmp_var_bound_strip_lb:
                var.setlb(instance._tmp_var_bound_strip_lb[var])
            if var in instance._tmp_var_bound_strip_ub:
                var.setub(instance._tmp_var_bound_strip_ub[var])
            if var in instance._tmp_var_bound_strip_domain:
                var.domain = instance._tmp_var_bound_strip_domain[var]
        del instance._tmp_var_bound_strip_lb
        del instance._tmp_var_bound_strip_ub
        del instance._tmp_var_bound_strip_domain
