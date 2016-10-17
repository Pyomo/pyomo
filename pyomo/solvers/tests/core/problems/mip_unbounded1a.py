from pyomo.core import *

model = ConcreteModel()

model.x = Var(within=Integers)
model.y = Var(within=Integers)

model.o = Objective(expr=model.x+model.y)
