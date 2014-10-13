from pyomo.environ import *

model = AbstractModel()
model.x = Var()

def c_rule(model):
    return model.x >= 10.0
model.c = Constraint(rule=c_rule)
