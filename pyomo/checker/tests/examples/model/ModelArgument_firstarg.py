from pyomo.environ import *

model = ConcreteModel()
model.S = RangeSet(10)
model.X = Var(model.S)

def C_rule(m, i):
    return m.X[i] >= 10.0
model.C = Constraint(rule=C_rule)
