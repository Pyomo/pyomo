#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.deprecation import deprecated
from pyomo.common.plugin import Plugin, implements
from pyomo.core import IPyomoScriptModifyInstance, TransformationFactory

# This import ensures that gdp.chull is registered, even if pyomo.environ
# was never imported.
import pyomo.gdp.plugins.chull

@deprecated('The GDP Pyomo script plugins are deprecated.  '
            'Use BuildActions or the --transform option.',
            version='5.4')
class ConvexHull_Transformation_PyomoScript_Plugin(Plugin):
    """Plugin to automatically call the GDP Convex Hull relaxation within
    the Pyomo script.

    """

    implements(IPyomoScriptModifyInstance, service=True)

    def apply(self, **kwds):
        instance = kwds.pop('instance')
        # Not sure why the ModifyInstance callback started passing the
        # model along with the instance.  We will ignore it.
        model = kwds.pop('model', None)
        xform = TransformationFactory('gdp.chull')
        return xform.apply_to(instance, **kwds)


transform = ConvexHull_Transformation_PyomoScript_Plugin()
