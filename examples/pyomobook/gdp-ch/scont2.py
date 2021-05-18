import pyomo.environ as pyo

import scont
model = scont.model

# @action:
def transform_gdp(m):
   xfrm = pyo.TransformationFactory('gdp.bigm')
   xfrm.apply_to(m)
model.transform_gdp = pyo.BuildAction(rule=transform_gdp)
# @:action
