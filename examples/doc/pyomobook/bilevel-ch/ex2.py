from pyomo.environ import *

from bard511 import M as model

# @transform:
xfrm = TransformationFactory('bilevel.linear_mpec')
xfrm.apply_to(model)
# @:transform

model.pprint()
