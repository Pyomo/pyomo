#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import *
import sys
import importlib
example = importlib.import_module(sys.argv[1])

xfrm = TransformationFactory('bilevel.linear_dual')
model2 = xfrm.apply(example.pyomo_create_model(None,None))
model2.pprint()

xfrm = TransformationFactory('gdp.bilinear')
model3 = xfrm.apply(model2)
model3.pprint()

