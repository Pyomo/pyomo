from coopr.pyomo import *

model = AbstractModel()
model.S = RangeSet(10)
model.X = Var(model.S)

if sum(model.X[i] for i in model.S) <= 10.0:
    pass
if sum(value(model.X[i]) for i in model.S) <= 10.0:
    pass
