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

from pyomo.core.base import (Transformation,
                             TransformationFactory,
                             Constraint,
                             Block,
                             SortComponents)
from pyomo.mpec.complementarity import Complementarity

from six import iterkeys

logger = logging.getLogger('pyomo.core')


#
# This transformation reworks each Complementarity block to 
# setup a standard form.
#
@TransformationFactory.register('mpec.standard_form', doc="Standard reformulation of complementarity condition")
class MPEC3_Transformation(Transformation):


    def __init__(self):
        super(MPEC3_Transformation, self).__init__()

    def _apply_to(self, instance, **kwds):
        options = kwds.pop('options', {})
        #
        # Iterate over the model finding Complementarity components
        #
        for block in instance.block_data_objects(active=True, sort=SortComponents.deterministic):
            for complementarity in block.component_objects(Complementarity, active=True, descend_into=False):
                for index in sorted(iterkeys(complementarity)):
                    _data = complementarity[index]
                    if not _data.active:
                        continue
                    _data.to_standard_form()
                    #
                block.reclassify_component_type(complementarity, Block)
