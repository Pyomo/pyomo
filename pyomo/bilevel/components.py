#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['SubModel']

from pyomo.core.base.component import Component, register_component
from pyomo.core.base.block import SimpleBlock

# TODO: Do we need to have SimpleSubModel and IndexedSubModel classes?


class SubModel(SimpleBlock):

    def __init__(self, *args, **kwargs):
        """Constructor"""
        #
        # Collect kwargs for SubModel
        #
        _rule = kwargs.pop('rule', None )
        _fixed = kwargs.pop('fixed', None )
        _var = kwargs.pop('var', None )     # Not documented
        #
        # Initialize the SimpleBlock
        #
        kwargs.setdefault('ctype', SubModel)
        SimpleBlock.__init__(self, *args, **kwargs)
        #
        # Initialize from kwargs
        #
        self._rule = _rule
        if isinstance(_fixed, Component):
            self._fixed = [_fixed]
        else:
            self._fixed = _fixed
        if isinstance(_var, Component):
            self._var = [_var]
        else:
            self._var = _var

register_component(SubModel, "A submodel in a bilevel program")
