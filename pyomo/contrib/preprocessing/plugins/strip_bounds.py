"""Transformation to strip variable bounds from a model."""
import textwrap

from pyomo.core.base.plugin import TransformationFactory
from pyomo.core.base.var import Var
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.set_types import Reals
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation
from pyomo.common.config import ConfigBlock, ConfigValue, add_docstring_list


@TransformationFactory.register('contrib.strip_var_bounds',
          doc="Strip bounds from varaibles.")
class VariableBoundStripper(NonIsomorphicTransformation):
    """Strip bounds from variables.

    Keyword arguments below are specified for the ``apply_to`` and
    ``create_using`` functions.

    """

    CONFIG = ConfigBlock()
    CONFIG.declare("strip_domains", ConfigValue(
        default=True, domain=bool,
        description="strip the domain for discrete variables as well"
    ))
    CONFIG.declare("reversible", ConfigValue(
        default=False, domain=bool,
        description="Whether the bound stripping will be temporary. "
        "If so, store information for reversion."
    ))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds)
        if config.reversible:
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
            if config.strip_domains and not var.domain == Reals:
                if config.reversible:
                    instance._tmp_var_bound_strip_domain[var] = var.domain
                var.domain = Reals
            if var.has_lb():
                if config.reversible:
                    instance._tmp_var_bound_strip_lb[var] = var.lb
                var.setlb(None)
            if var.has_ub():
                if config.reversible:
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
