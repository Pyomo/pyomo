from coopr.pyomo import *

model = AbstractModel()
model.S = RangeSet(10)
model.X = Var(model.S)

def c_rule(m, i):
    if sum(m.X[i] for i in m.S) <= 10.0:
        pass
    if sum(value(m.X[i]) for i in m.S) <= 10.0:
        pass
    return sum(m.X[i] for i in m.S) <= 10.0
model.C = Constraint(rule=c_rule)
