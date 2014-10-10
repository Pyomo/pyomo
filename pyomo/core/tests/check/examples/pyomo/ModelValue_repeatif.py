from coopr.pyomo import *

model = AbstractModel()
model.X = Var()

if model.X >= 10.0:
    pass
if value(model.X) >= 10.0:
    pass
if model.X >= 10.0:
    pass

if model.X >= 10.0:
    if model.X >= 10.0:
        pass
