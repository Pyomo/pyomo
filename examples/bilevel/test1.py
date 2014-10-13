from pyomo.environ import *
import sys
import importlib
example = importlib.import_module(sys.argv[1])

xfrm = TransformationFactory('bilevel.linear_dual')
model2 = xfrm.apply(example.pyomo_create_model(None,None))
model2.pprint()

xfrm = TransformationFactory('gdp.bilinear')
model3 = xfrm.apply(model2)
model3.pprint()

