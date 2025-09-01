#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

logger = logging.getLogger('pyomo.core')

from pyomo.common import deprecated
from pyomo.common.config import ConfigDict, ConfigValue, In, IsInstance
from pyomo.common.deprecation import deprecation_warning
from pyomo.core.base import (
    Transformation,
    TransformationFactory,
    Var,
    Suffix,
    Reals,
    Block,
    ReverseTransformationToken,
    VarCollector,
    Constraint,
    Objective,
)
from pyomo.core.base.block import SubclassOf
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct
from pyomo.util.vars_from_expressions import get_vars_from_components


#
# This transformation relaxes integer ranges to their continuous
# counterparts
#
@TransformationFactory.register(
    'core.relax_integer_vars', doc="Relax integer variables to continuous counterparts"
)
class RelaxIntegerVars(Transformation):
    CONFIG = ConfigDict('core.relax_integer_vars')
    CONFIG.declare(
        'targets',
        ConfigValue(
            default=None,
            domain=target_list,
            description="target or list of targets that will be relaxed",
            doc="""
            This specifies the list of components to relax. If None (default), the
            entire model is transformed. Note that if the transformation is done
            out of place, the list of targets should be attached to the model before
            it is cloned, and the list will specify the targets on the cloned
            instance.""",
        ),
    )
    CONFIG.declare(
        'reverse',
        ConfigValue(
            default=None,
            domain=IsInstance(ReverseTransformationToken),
            description="The token returned by a (forward) call to this "
            "transformation, if you wish to reverse the transformation.",
            doc="""
            This argument should be the reverse transformation token
            returned by a previous call to this transformation to transform
            fixed discrete state in the given model.
            If this argument is specified, this call to the transformation
            will reverse what the transformation did in the call that returned
            the token. Note that if there are intermediate changes to the model
            in between the forward and the backward calls to the transformation,
            the behavior could be unexpected.
            """,
        ),
    )
    CONFIG.declare(
        'var_collector',
        ConfigValue(
            default=VarCollector.FromVarComponents,
            domain=In(VarCollector),
            description="The method for collection the Vars to relax. If "
            "VarCollector.FromVarComponents (default), any Var component on "
            "the active tree will be relaxed.",
            doc="""
            This specifies the method for collecting the Var components to relax.
            The default, VarCollector.FromVarComponents, assumes that all relevant
            Vars are on the active tree. If this is true, then this is the most
            performant option. However, in more complex cases where some Vars may not
            be in the active tree (e.g. some are on deactivated Blocks or come from
            other models), specify VarCollector.FromExpressions to relax all Vars that
            appear in expressions in the active tree.
            """,
        ),
    )
    CONFIG.declare(
        'transform_deactivated_blocks',
        ConfigValue(
            default=True,
            description="[DEPRECATED]: Whether or not to search for Var components to "
            "relax on deactivated Blocks. True by default",
        ),
    )
    CONFIG.declare(
        'undo',
        ConfigValue(
            default=False,
            domain=bool,
            description="[DEPRECATED]: Please use the 'reverse' argument to undo "
            "the transformation.",
        ),
    )

    def __init__(self):
        super().__init__()

    def _apply_to(self, model, **kwds):
        if model.ctype not in SubclassOf(Block):
            raise ValueError(
                "Transformation called on %s of type %s. 'model' "
                "must be a ConcreteModel or Block." % (model.name, model.ctype)
            )
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        if config.undo:
            deprecation_warning(
                "The 'undo' argument is deprecated. Please use the 'reverse' "
                "argument to undo the transformation.",
                version='6.9.3',
            )
            for v, d in model._relaxed_integer_vars[None].values():
                bounds = v.bounds
                v.domain = d
                v.setlb(bounds[0])
                v.setub(bounds[1])
            model.del_component("_relaxed_integer_vars")
            return

        targets = (model,) if config.targets is None else config.targets

        if config.reverse is None:
            reverse_dict = {}
            # Relax the model
            reverse_token = ReverseTransformationToken(
                self.__class__, model, targets, reverse_dict
            )
        else:
            # reverse the transformation
            reverse_token = config.reverse
            reverse_token.check_token_valid(self.__class__, model, targets)
            reverse_dict = reverse_token.reverse_dict
            for v, d in reverse_dict.values():
                lb, ub = v.bounds
                v.domain = d
                v.setlb(lb)
                v.setub(ub)
            return

        ### [ESJ 4/29/25]: This can go away when we remove 'undo'
        model._relaxed_integer_vars = Suffix(direction=Suffix.LOCAL)
        model._relaxed_integer_vars[None] = reverse_dict
        ###

        for t in targets:
            if isinstance(t, Block):
                blocks = t.values() if t.is_indexed() else (t,)
                for block in blocks:
                    self._relax_block(block, config, reverse_dict)
            elif t.ctype is Var:
                self._relax_var(t, reverse_dict)
            else:
                raise ValueError(
                    "Target '%s' was not a Block or Var. It was of type "
                    "'%s' and cannot be transformed." % (t.name, type(t))
                )

        return reverse_token

    def _relax_block(self, block, config, reverse_dict):
        self._relax_vars_from_block(block, config, reverse_dict)

        for b in block.component_data_objects(Block, active=None, descend_into=True):
            if not b.active:
                if config.transform_deactivated_blocks:
                    deprecation_warning(
                        "The `transform_deactivated_blocks` arguments is deprecated. "
                        "Either specify deactivated Blocks as targets to activate them "
                        "if transforming them is the desired behavior.",
                        version='6.9.3',
                    )
                else:
                    continue
            self._relax_vars_from_block(b, config, reverse_dict)

    def _relax_vars_from_block(self, block, config, reverse_dict):
        if config.var_collector is VarCollector.FromVarComponents:
            model_vars = block.component_data_objects(Var, descend_into=False)
        else:
            model_vars = get_vars_from_components(
                block, ctype=(Constraint, Objective), descend_into=False
            )
        for var in model_vars:
            if id(var) not in reverse_dict:
                self._relax_var(var, reverse_dict)

    def _relax_var(self, v, reverse_dict):
        var_datas = v.values() if v.is_indexed() else (v,)
        for var in var_datas:
            if not var.is_integer():
                continue
            lb, ub = var.bounds
            _domain = var.domain
            var.domain = Reals
            var.setlb(lb)
            var.setub(ub)
            reverse_dict[id(var)] = (var, _domain)


@TransformationFactory.register(
    'core.relax_discrete',
    doc="[DEPRECATED] Relax integer variables to continuous counterparts",
)
@deprecated(
    "core.relax_discrete is deprecated.  Use core.relax_integer_vars", version='5.7'
)
class RelaxDiscreteVars(RelaxIntegerVars):
    """
    This plugin relaxes integrality in a Pyomo model.
    """

    def __init__(self, **kwds):
        super(RelaxDiscreteVars, self).__init__(**kwds)


#
# This transformation fixes known discrete domains to their current values
#
@TransformationFactory.register(
    'core.fix_integer_vars', doc="Fix all integer variables to their current values"
)
class FixIntegerVars(Transformation):
    def __init__(self):
        super(FixIntegerVars, self).__init__()

    def _apply_to(self, model, **kwds):
        options = kwds.pop('options', {})
        if kwds.get('undo', options.get('undo', False)):
            for v in model._fixed_integer_vars[None]:
                v.unfix()
            model.del_component("_fixed_integer_vars")
            return

        fixed_vars = []
        _base_model_vars = model.component_data_objects(
            Var, active=True, descend_into=True
        )
        for var in _base_model_vars:
            # Instead of checking against
            # `_integer_relaxation_map.keys()` we just check the item
            # properties to fix #995 When #326 has been resolved, we can
            # check against the dict-keys again
            if var.is_integer() and not var.is_fixed():
                fixed_vars.append(var)
                var.fix()
        model._fixed_integer_vars = Suffix(direction=Suffix.LOCAL)
        model._fixed_integer_vars[None] = fixed_vars


@TransformationFactory.register(
    'core.fix_discrete',
    doc="[DEPRECATED] Fix all integer variables to their current values",
)
@deprecated(
    "core.fix_discrete is deprecated.  Use core.fix_integer_vars", version='5.7'
)
class FixDiscreteVars(FixIntegerVars):
    def __init__(self, **kwds):
        super(FixDiscreteVars, self).__init__(**kwds)
