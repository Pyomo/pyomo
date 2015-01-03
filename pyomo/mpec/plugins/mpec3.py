#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from six import iterkeys
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation, Var, Constraint, Objective, active_components, Block, SortComponents
from pyomo.mpec.complementarity import Complementarity

import logging
logger = logging.getLogger('pyomo.core')


#
# This transformation reworks each Complementarity block to 
# setup a standard form.
#
class MPEC3_Transformation(Transformation):

    alias('mpec.standard_form', doc="Standard reformulation of complementarity condition")

    def __init__(self):
        super(MPEC3_Transformation, self).__init__()

    def apply(self, instance, **kwds):
        options = kwds.pop('options', {})
        #
        # Iterate over the model finding Complementarity components
        #
        for block in instance.all_blocks(active=True, sort=SortComponents.deterministic):
            for complementarity in active_components(block,Complementarity):
                for index in sorted(iterkeys(complementarity)):
                    _data = complementarity[index]
                    if not _data.active:
                        continue
                    _data.to_standard_form()
                    #
                block.reclassify_component_type(complementarity, Block)
        #
        instance.preprocess()
        return instance

