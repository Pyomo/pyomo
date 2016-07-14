from pyomo.environ import *

model = AbstractModel()
model.x = Var()
model.p = Param(mutable=True, initialize=1.0)
def cost_rule(model, i):
    if i == 1:
        return model.x
    else:
        return 0.0
model.cost = Expression([1,2], rule=cost_rule)
def o_rule(model):
    return model.x
model.o = Objective(rule=o_rule)
def c_rule(model):
    return model.x >= model.p
model.c = Constraint(rule=c_rule)
