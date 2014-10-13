from pyomo.environ import *

model = AbstractModel()
model.X = Var()

if model.X >= 10.0:
    pass
if value(model.X) >= 10.0:
    pass
