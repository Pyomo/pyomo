#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

from six import iterkeys
from pyomo.misc.plugin import alias
from pyomo.core.base import Transformation, Var, Constraint, Objective, active_components, Block, Param
from pyomo.mpec.complementarity import Complementarity

import logging
logger = logging.getLogger('pyomo.core')


#
# This transformation reworks each Complementarity block to 
# add a constraint that ensures the complementarity condition.
# Specifically, 
#
#   x1 >= 0  OR  x2 >= 0
#
# becomes
#
#   x1 >= 0
#   x2 >= 0
#   x1*x2 <= 0
#
class MPEC1_Transformation(Transformation):

    alias('mpec.simple_nonlinear', doc="Nonlinear transformations of complementarity conditions when all variables are non-negative")

    def __init__(self):
        super(MPEC1_Transformation, self).__init__()

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})
        #
        # Create a mutable parameter that defines the value of the upper bound
        # on the constraints
        #
        instance.mpec_bound = Param(mutable=True, initialize=0)
        #
        # Iterate over the model finding Complementarity components
        #
        for block in instance.all_blocks(sort_by_keys=True):
            for complementarity in active_components(block,Complementarity):
                for index in sorted(iterkeys(complementarity)):
                    _data = complementarity[index]
                    if not _data.active:
                        continue
                    _data.to_standard_form()
                    #
                    _type = getattr(_data.c, "_complementarity", 0)
                    if _type == 1:
                        #
                        # Variable v is bounded below, so we can replace 
                        # constraint c with a constraint that ensure that either
                        # constraint c is active or variable v is at its lower bound.
                        #
                        _data.ccon = Constraint(expr=(_data.c.body - _data.c.lower)*_data.v <= instance.mpec_bound)
                        _data.c.deactivate()
                    elif _type == 2:
                        #
                        # Variable v is bounded above, so we can replace 
                        # constraint c with a constraint that ensure that either
                        # constraint c is active or variable v is at its upper bound.
                        #
                        _data.ccon = Constraint(expr=(_data.c.body - _data.c.lower)*_data.v <= instance.mpec_bound)
                        _data.c.deactivate()
                    elif _type == 3:
                        #
                        # Variable v is bounded above and below.  We can define
                        #
                        _data.ccon_l = Constraint(expr=(_data.v - _data.v.bounds[0])*_data.c.body <= instance.mpec_bound)
                        _data.ccon_u = Constraint(expr=(_data.v - _data.v.bounds[1])*_data.c.body <= instance.mpec_bound)
                        _data.c.deactivate()
                block.reclassify_component_type(complementarity, Block)
        #
        instance.preprocess()
        return instance

