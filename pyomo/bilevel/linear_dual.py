#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

from coopr.core.plugin import Plugin, implements
from coopr.pyomo import *


class LinearDual_Transformation_Plugin(Plugin):

    implements(IPyomoScriptModifyInstance)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        xform = TransformationFactory('bilevel.linear_dual')
        return xform.apply(instance, **kwds)


transform = LinearDual_Transformation_Plugin()
transform.activate()

