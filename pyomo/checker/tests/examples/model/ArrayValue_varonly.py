from pyomo.environ import *

model = AbstractModel()
model.w = Var(within=NonNegativeReals)
model.x = Var()

model.w.value = 42
model.x.value = 42
