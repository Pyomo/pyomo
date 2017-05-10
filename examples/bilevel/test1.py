#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
import sys
import pyutilib.misc
pyutilib.misc.import_file(sys.argv[1])

xfrm = TransformationFactory('bilevel.linear_dual')
model2 = xfrm.apply(example.pyomo_create_model(None,None))
model2.pprint()

xfrm = TransformationFactory('gdp.bilinear')
model3 = xfrm.apply(model2)
model3.pprint()

