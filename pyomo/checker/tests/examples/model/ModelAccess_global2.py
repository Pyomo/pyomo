from pyomo.environ import *

model = ConcreteModel()
model.X = Var()

def c_rule(m):
    try:
        return model.X >= 10.0
    except Exception:
        return model.X >= 20.0
model.C = Constraint(rule=c_rule)
