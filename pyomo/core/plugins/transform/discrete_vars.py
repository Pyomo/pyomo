#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import logging
logger = logging.getLogger('pyomo.core')

from six import itervalues

from pyomo.util.plugin import alias
from pyomo.core.base import ( 
    Transformation,
    Binary,
    Boolean,
    Integers,
    Reals, 
    PositiveIntegers,
    PositiveReals, 
    NonPositiveIntegers,
    NonPositiveReals, 
    NegativeIntegers,
    NegativeReals,
    NonNegativeIntegers,
    NonNegativeReals,
    IntegerInterval,
    RealInterval,
    Var,
    Suffix,
)

_discrete_relaxation_map = {
    Binary : NonNegativeReals,
    Boolean : NonNegativeReals,
    Integers : Reals, 
    PositiveIntegers : PositiveReals, 
    NonPositiveIntegers : NonPositiveReals, 
    NegativeIntegers : NegativeReals,
    NonNegativeIntegers : NonNegativeReals,
    IntegerInterval : RealInterval,
}


#
# This transformation relaxes known discrete domains to their continuous
# counterparts
#
class RelaxDiscreteVars(Transformation):

    alias( 'core.relax_discrete', 
           doc="Relax known discrete domains to continuous counterparts" )

    def __init__(self):
        super(RelaxDiscreteVars, self).__init__()

    def _apply_to(self, model, **kwds): 
        options = kwds.pop('options', {})
        if kwds.get('undo', options.get('undo', False)):
            for v, d in itervalues(model._relaxed_discrete_vars[None]):
                v.domain = d
            model.del_component("_relaxed_discrete_vars")
            return
        
        # Relax the model
        relaxed_vars = {}
        _base_model_vars = model.component_data_objects(
            Var, active=True, descend_into=True )
        for var in _base_model_vars:
            if var.domain in _discrete_relaxation_map:
                if var.domain is Binary or var.domain is Boolean:
                    var.setlb(0)
                    var.setub(1)
                # Note: some indexed components can only have their
                # domain set on the parent component (the individual
                # indices cannot be set independently)
                _c = var.parent_component()
                if id(_c) in _discrete_relaxation_map:
                    continue
                try:
                    _domain = var.domain
                    var.domain = _discrete_relaxation_map[_domain]
                    relaxed_vars[id(var)] = (var, _domain)
                except:
                    _domain = _c.domain
                    _c.domain = _discrete_relaxation_map[_domain]
                    relaxed_vars[id(_c)] = (_c, _domain)
        model._relaxed_discrete_vars = Suffix(direction=Suffix.LOCAL)
        model._relaxed_discrete_vars[None] = relaxed_vars


#
# This transformation fixes known discrete domains to their current values
#
class FixDiscreteVars(Transformation):

    alias( 'core.fix_discrete', 
           doc="Fix known discrete domains to continuous counterparts" )

    def __init__(self):
        super(FixDiscreteVars, self).__init__()

    def _apply_to(self, model, **kwds): 
        options = kwds.pop('options', {})
        if kwds.get('undo', options.get('undo', False)):
            for v in model._fixed_discrete_vars[None]:
                v.unfix()
            model.del_component("_fixed_discrete_vars")
            return
        
        fixed_vars = []
        _base_model_vars = model.component_data_objects(
            Var, active=True, descend_into=True )
        for var in _base_model_vars:
            if var.domain in _discrete_relaxation_map and not var.is_fixed():
                fixed_vars.append(var)
                var.fix()
        model._fixed_discrete_vars = Suffix(direction=Suffix.LOCAL)
        model._fixed_discrete_vars[None] = fixed_vars

