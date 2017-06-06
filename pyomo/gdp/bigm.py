#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.util.plugin import Plugin, implements
from pyomo.core import *


class BigM_Transformation_Plugin(Plugin):

    implements(IPyomoScriptModifyInstance, service=True)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('gdp.bigm')
        return xform.apply(instance, **kwds)


transform = BigM_Transformation_Plugin()
