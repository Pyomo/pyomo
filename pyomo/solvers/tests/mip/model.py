from pyomo.core import *

model = AbstractModel()

model.A = RangeSet(1,4)

model.x = Var(model.A)

def obj_rule(model):
    return summation(model.x)
model.obj = Objective(rule=obj_rule)
