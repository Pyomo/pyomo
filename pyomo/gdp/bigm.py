#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.util.plugin import Plugin, implements
from pyomo.core import *


class BigM_Transformation_Plugin(Plugin):

    implements(IPyomoScriptModifyInstance, service=True)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('gdp.bigm')
        return xform.apply(instance, **kwds)


transform = BigM_Transformation_Plugin()
