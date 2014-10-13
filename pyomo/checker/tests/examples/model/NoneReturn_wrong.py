from pyomo.environ import *

model = AbstractModel()
model.X = Var()

def c_rule(m):
    return None
model.C = Constraint(rule=c_rule)
