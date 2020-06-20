#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
logger = logging.getLogger('pyomo.core')

from six import itervalues

from pyomo.common import deprecated
from pyomo.core.base import (
    Transformation,
    TransformationFactory,
    Var,
    Suffix,
    Reals,
)

#
# This transformation relaxes integer ranges to their continuous
# counterparts
#
@TransformationFactory.register(
    'core.relax_integer_vars',
    doc="Relax integer variables to continuous counterparts" )
class RelaxIntegerVars(Transformation):

    def __init__(self):
        super(RelaxIntegerVars, self).__init__()

    def _apply_to(self, model, **kwds):
        options = kwds.pop('options', {})
        if kwds.get('undo', options.get('undo', False)):
            for v, d in itervalues(model._relaxed_integer_vars[None]):
                bounds = v.bounds
                v.domain = d
                v.setlb(bounds[0])
                v.setub(bounds[1])
            model.del_component("_relaxed_integer_vars")
            return
        # True by default, you can specify False if you want
        descend = kwds.get('transform_deactivated_blocks',
                           options.get('transform_deactivated_blocks', True))
        active = None if descend else True

        # Relax the model
        relaxed_vars = {}
        _base_model_vars = model.component_data_objects(
            Var, active=active, descend_into=True )
        for var in _base_model_vars:
            if not var.is_integer():
                continue
            # Note: some indexed components can only have their
            # domain set on the parent component (the individual
            # indices cannot be set independently)
            _c = var.parent_component()
            try:
                lb, ub = var.bounds
                _domain = var.domain
                var.domain = Reals
                var.setlb(lb)
                var.setub(ub)
                relaxed_vars[id(var)] = (var, _domain)
            except:
                if id(_c) in relaxed_vars:
                    continue
                _domain = _c.domain
                lb, ub = _c.bounds
                _c.domain = Reals
                _c.setlb(lb)
                _c.setub(ub)
                relaxed_vars[id(_c)] = (_c, _domain)
        model._relaxed_integer_vars = Suffix(direction=Suffix.LOCAL)
        model._relaxed_integer_vars[None] = relaxed_vars


@TransformationFactory.register(
    'core.relax_discrete',
    doc="[DEPRECATED] Relax integer variables to continuous counterparts" )
class RelaxDiscreteVars(RelaxIntegerVars):
    """
    This plugin relaxes integrality in a Pyomo model.
    """

    @deprecated(
        "core.relax_discrete is deprecated.  Use core.relax_integer_vars",
        version='5.7')
    def __init__(self, **kwds):
        super(RelaxDiscreteVars, self).__init__(**kwds)


#
# This transformation fixes known discrete domains to their current values
#
@TransformationFactory.register(
    'core.fix_integer_vars',
    doc="Fix all integer variables to their current values")
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
            Var, active=True, descend_into=True)
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
    doc="[DEPRECATED] Fix all integer variables to their current values")
class FixDiscreteVars(FixIntegerVars):
    @deprecated(
        "core.fix_discrete is deprecated.  Use core.fix_integer_vars",
        version='5.7')
    def __init__(self, **kwds):
        super(FixDiscreteVars, self).__init__(**kwds)
