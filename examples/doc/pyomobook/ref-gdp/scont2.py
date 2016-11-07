from pyomo.environ import *

import scont
model = scont.model

# @action:
def transform_gdp(m):
   xfrm = TransformationFactory('gdp.bigm')
   xfrm.apply_to(m)
model.transform_gdp = BuildAction(rule=transform_gdp)
# @:action
