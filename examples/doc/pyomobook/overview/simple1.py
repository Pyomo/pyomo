import pyomo.environ
from pyomo.core import *

# @body:
model = AbstractModel()
model.I = Set()
model.p = Param(model.I)
# @:body
