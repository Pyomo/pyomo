from pyomo.environ import *

from bard511 import model

# @transform:
xfrm = TransformationFactory('bilevel.linear_mpec')
xfrm.apply_to(model)
# @:transform
