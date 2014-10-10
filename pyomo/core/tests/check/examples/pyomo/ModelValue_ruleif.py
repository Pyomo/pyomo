from pyomo.core import *

model = AbstractModel()
model.X = Var()

def c_rule(m):
    if m.X >= 10.0:
        pass
    if value(m.X) >= 10.0:
        pass
    return m.X >= 10.0

model.C = Constraint(rule=c_rule)
