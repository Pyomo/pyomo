from pyomo.environ import *

model = ConcreteModel()
model.X = Var()

def c_rule(m):
    return model.X >= 10.0 # wrongly access global 'model'
model.C = Constraint(rule=c_rule)
