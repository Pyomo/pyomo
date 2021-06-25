#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['SubModel']

from pyomo.core.base.block import ScalarBlock
from pyomo.core.base.component import Component, ModelComponentFactory
from pyomo.common.deprecation import deprecated

# TODO: Do we need to have ScalarSubModel and IndexedSubModel classes?


@ModelComponentFactory.register("A submodel in a bilevel program")
class SubModel(ScalarBlock):

    @deprecated(
        "Use of the pyomo.bilevel package is deprecated. There are known bugs "
        "in pyomo.bilevel, and we do not recommend the use of this code. "
        "Development of bilevel optimization capabilities has been shifted to "
        "the Pyomo Adversarial Optimization (PAO) library. Please contact "
        "William Hart for further details (wehart@sandia.gov).",
        version='5.6.2')
    def __init__(self, *args, **kwargs):
        """Constructor"""
        #
        # Collect kwargs for SubModel
        #
        _rule = kwargs.pop('rule', None )
        _fixed = kwargs.pop('fixed', None )
        _var = kwargs.pop('var', None )     # Not documented
        #
        # Initialize the ScalarBlock
        #
        kwargs.setdefault('ctype', SubModel)
        ScalarBlock.__init__(self, *args, **kwargs)
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

