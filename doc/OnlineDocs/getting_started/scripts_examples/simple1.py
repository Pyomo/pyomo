from __future__ import division
from pyomo.environ import *

# @body:
model = AbstractModel()
model.I = Set()
model.p = Param(model.I)
# @:body
