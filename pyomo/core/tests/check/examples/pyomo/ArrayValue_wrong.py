from coopr.pyomo import *

model = AbstractModel()
model.y = Var([10])
model.s = RangeSet(10)
model.z = Var(model.s)

model.y.value = 42
model.z.value = 42
