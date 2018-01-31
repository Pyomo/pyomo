
from pyomo.util.plugin import alias
from pyomo.core.base import Transformation


class Xfrm_PyomoTransformation(Transformation):

    alias('contrib.example.xfrm', doc="An example of a transformation in a pyomo.contrib package")

    def __init__(self):
        super(Xfrm_PyomoTransformation, self).__init__()

    def create_using(self, instance, **kwds):
        # This transformation doesn't do anything...
        return instance

