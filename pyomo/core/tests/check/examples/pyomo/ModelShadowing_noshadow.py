from coopr.pyomo import *

model = AbstractModel()
model.x = Var()

def c_rule(m):
    return m.x >= 10.0
model.c = Constraint(rule=c_rule)
