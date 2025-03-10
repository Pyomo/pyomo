#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import Transformation, TransformationFactory


@TransformationFactory.register(
    'contrib.example.xfrm',
    doc="An example of a transformation in a pyomo.contrib package",
)
class Xfrm_PyomoTransformation(Transformation):
    def __init__(self):
        super(Xfrm_PyomoTransformation, self).__init__()

    def create_using(self, instance, **kwds):
        # This transformation doesn't do anything...
        return instance
